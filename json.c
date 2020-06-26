#include "json.h"

#include <string.h>
#include <stdint.h>
#include <setjmp.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif
#ifdef __SSE2__
#include <pmmintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif


// uncomment to enable reentrancy in parse_loop
//#define JSON_STACK_WORKING_MEMORY


#define JSON_ALLOC(buf_name, new_len) \
    if (context->buf_name##_brk + new_len >= context->buf_name##_size) { \
        if (!context->buf_name##_realloc) \
            error("out of " #buf_name " memory", ptr, context); \
        context->buf_name = context->buf_name##_realloc(context->buf_name, context->buf_name##_brk + new_len); \
        if (!context->buf_name) \
            error("out of " #buf_name " memory", ptr, context);} \
        context->buf_name##_brk += new_len;
#define JSON_ALIGN(buf_name) \
        context->buf_name##_brk = context->buf_name##_brk + (-context->buf_name##_brk & 0x1F); // align to 32byte boundary for AVX

typedef struct json_context_impl_t
{
    const char* source;
    char* string_buffer;
    unsigned int string_buffer_size;
    unsigned int string_buffer_brk;
    realloc_callback_t* string_buffer_realloc;
    json_key_value_t* key_val_buffer;
    unsigned int key_val_buffer_size;
    unsigned int key_val_buffer_brk;
    realloc_callback_t* key_val_buffer_realloc;
    json_result_t result;
    int nest_depth;
    jmp_buf error_jmp_buf;
} json_context_impl_t;

_Noreturn static inline void error(const char* reason, const char** ptr, json_context_impl_t* context)
{
    context->result.error.reason = reason;
    context->result.error.index = *ptr - context->source;
    longjmp(context->error_jmp_buf, 1);
}

static inline void expect(const char** ptr, char c, json_context_impl_t* context)
{
    if (__builtin_expect(**ptr != c, 0))
    {
        error("unexpected token found", ptr, context);
    }
    else
    {
        ++*ptr;
    }
}
static inline bool accept(const char** ptr, char c, json_context_impl_t* context)
{
    (void)context;
    if (**ptr == c)
    {
        ++*ptr;
        return true;
    }
    else
    {
        return false;
    }
}

static inline bool is_quote_escaped(const char* ptr)
{
    uint8_t backslash_count = 0;

    while (ptr[-backslash_count-1] == '\\')
        ++backslash_count;

    return backslash_count&1;
}


// TODO : test further in error paths to detect if the string was unterminated
static inline const char* find_end_of_string_lit(const char* str, bool* has_escaped_chars, json_context_impl_t* context, char* target)
{
#if defined(__AVX2__)
    for(;;str += 32)
    {
        // first test for the presence of quotes with 32 bytes vectors
        for (;;str += 32)
        {
            const __m256i data = _mm256_loadu_si256((const __m256i*)str);
            // speculative write
            _mm256_storeu_si256((__m256i*)target, data); target += 32;

            const __m256i quote_mask        = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('"'));
            //quotes!
            if (!_mm256_testz_si256(quote_mask, quote_mask))
                break;
            else
            {
                const __m256i invalid_char_mask = _mm256_cmpeq_epi8(data, _mm256_min_epu8(data, _mm256_set1_epi8(0x1A))); // no unsigned 8-bit cmplt
                if (!_mm256_testz_si256(invalid_char_mask, invalid_char_mask))
                    error("control characters must be escaped", &str, context);
            }
        }


        // weird case first
        if (str[0] == '"')
        {
            if (!is_quote_escaped(&str[0]))
                return str;
            ++str;
        }

        const __m256i data = _mm256_loadu_si256((const __m256i*)str);
        // speculative write
        _mm256_storeu_si256((__m256i*)target, data); target += 32;

        const __m256i quote_mask        = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('"'));

        const __m256i backslash_mask    = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\\'));
        const __m256i invalid_char_mask = _mm256_cmpeq_epi8(data, _mm256_min_epu8(data, _mm256_set1_epi8(0x1A))); // no unsigned 8-bit cmplt
        const uint32_t invalid_char_movemask = _mm256_movemask_epi8(invalid_char_mask);
        const uint32_t backslash_movemask = _mm256_movemask_epi8(backslash_mask);
        uint32_t quote_movemask = _mm256_movemask_epi8(quote_mask);

        const uint32_t lone_quotes = ~(backslash_movemask<<1) & quote_movemask;
        const uint32_t potentially_tricky_escaped_quotes = (backslash_movemask<<2) & ((quote_movemask&~lone_quotes));

        // no quotes at all
        if (__builtin_expect(!potentially_tricky_escaped_quotes, 1) && __builtin_expect(!lone_quotes, 1))
        {
            if (backslash_movemask != 0)
                *has_escaped_chars = true;

            if (__builtin_expect(invalid_char_movemask != 0, 0))
                error("control characters must be escaped", &str, context);
        }
        else if (__builtin_expect(lone_quotes, 1) && __builtin_expect(!potentially_tricky_escaped_quotes, 1))
        {
            const uint32_t idx = __builtin_ctz(lone_quotes);
            const uint32_t idx_mask = (uint64_t)(0xFFFFFFFF) >> (32-idx);

            if ((backslash_movemask&idx_mask) != 0)
                *has_escaped_chars = true;

            if (__builtin_expect((invalid_char_movemask&idx_mask) != 0, 0))
                error("control characters must be escaped", &str, context);

            str += idx;
            return str;
        }

        // tricky case where multiple backslashes are next to a quote, very rare
        else
        {
            // iterate over every possible index
            uint32_t idx = 0;
            while (quote_movemask != 0)
            {
                int offset = __builtin_ctz(quote_movemask);
                idx += offset;
                quote_movemask >>= offset;
                if (!is_quote_escaped(&str[idx])) // even amount of backslashes <=> unescaped quote
                {
                    const uint32_t idx_mask = (uint64_t)(0xFFFFFFFF) >> (32-idx);

                    if ((backslash_movemask&idx_mask) != 0)
                        *has_escaped_chars = true;

                    if ((invalid_char_movemask&idx_mask) != 0)
                        error("control characters must be escaped", &str, context);

                    str += idx;
                    return str;
                }
                quote_movemask >>= 1; ++idx;
            }
            // only escaped quotes here
            if ((invalid_char_movemask) != 0)
                error("control characters must be escaped", &str, context);
        }
    }
#elif defined(__SSE2__)
    for(;;str += 16)
    {
        const __m128i data = _mm_loadu_si128((const __m128i*)str);
        const __m128i backslash_mask    = _mm_cmpeq_epi8(data, _mm_set1_epi8('\\'));
        const __m128i quote_mask        = _mm_cmpeq_epi8(data, _mm_set1_epi8('"'));
        const __m128i invalid_char_mask = _mm_cmpeq_epi8(data, _mm_min_epu8(data, _mm_set1_epi8(0x1A))); // no unsigned 8-bit cmplt
        int quote_movemask = _mm_movemask_epi8(quote_mask);
        const int invalid_char_movemask = _mm_movemask_epi8(invalid_char_mask);
        const int backslash_movemask = _mm_movemask_epi8(backslash_mask);
        if (quote_movemask)
        {
            const int first_idx = __builtin_ctz(quote_movemask);
            // check that it's an actual quote, not an escaped quote
            if (str[-1] == '\\' || backslash_movemask != 0)
            {
                // iterate over every possible index
                uint32_t idx = first_idx;
                quote_movemask >>= idx;
                while (quote_movemask != 0)
                {
                    while ((quote_movemask&1) == 0)
                    {
                        ++idx;
                        quote_movemask >>= 1;
                    }
                    if (!is_quote_escaped(&str[idx])) // even amount of backslashes <=> unescaped quote
                    {
                        const uint16_t idx_mask = (0xFFFF) >> (16-idx);

                        if ((invalid_char_movemask&idx_mask) != 0)
                            error("control characters must be escaped", &str, context);

                        str += idx;
                        return str;
                    }
                    quote_movemask >>= 1; ++idx;
                }
                // only escaped quotes actually...
                if ((invalid_char_movemask) != 0)
                    error("control characters must be escaped", &str, context);
            }
            // no need to check, there aren't any backslashes in this chunk nor in the previous character
            else
            {
                const uint16_t idx_mask = (0xFFFF) >> (16-first_idx);

                if ((invalid_char_movemask&idx_mask) != 0)
                    error("control characters must be escaped", &str, context);

                str += first_idx;
                return str;
            }
        }
        else
        {
            if (invalid_char_movemask != 0)
                error("control characters must be escaped", &str, context);
        }
    }
#else
    const char* str_start = str;
    while (*str)
    {
        if (*(uint8_t*)str < 0x20) // unescaped control characted
        {
            error("control characters must be escaped", &str, context);
        }
        else if (str[0] == '"')
        {
            if (!is_quote_escaped(str)) // even amount of backslashes <=> unescaped quote
                return str;
        }
        ++str;
    }

    error("unexpected end of string", &str, context);
#endif
}

static void utf8_encode(uint16_t codepoint, char** ptr)
{
    if (codepoint < 0x80)
    {
        **ptr = (codepoint&0x7F);
        ++*ptr;
    }
    else if (codepoint < 0x0800)
    {
        **ptr = (codepoint >> 6 & 0x1F) | 0xC0;
        ++*ptr;
        **ptr = (codepoint & 0x3F) | 0x80;
        ++*ptr;
    }
    else
    {
        **ptr = (codepoint >> 12 & 0x0F) | 0xE0;
        ++*ptr;
        **ptr = (codepoint >> 6 & 0x3F) | 0x80;
        ++*ptr;
        **ptr = (codepoint & 0x3F) | 0x80;
        ++*ptr;
    }
}

    static uint8_t hex_digits[256] =
        {
            [0 ... 255] = 0xFF, // invalid
            ['0'] = 0,
            ['1'] = 1,
            ['2'] = 2,
            ['3'] = 3,
            ['4'] = 4,
            ['5'] = 5,
            ['6'] = 6,
            ['7'] = 7,
            ['8'] = 8,
            ['9'] = 9,
            ['a'] = 0xA, ['A'] = 0xA,
            ['b'] = 0xB, ['B'] = 0xB,
            ['c'] = 0xC, ['C'] = 0xC,
            ['d'] = 0xD, ['D'] = 0xD,
            ['e'] = 0xE, ['E'] = 0xE,
            ['f'] = 0xF, ['F'] = 0xF,
            };

static inline bool check_is_codepoint(const uint8_t* ptr)
{
    for (int i = 0; i < 4; ++i)
    {
        if (hex_digits[ptr[i]] == 0xFF)
            return false;
    }
    return true;
}

static uint16_t unescape_codepoint(const uint8_t* ptr)
{
    uint16_t codepoint;

    codepoint  = hex_digits[ptr[0]] << 12;
    codepoint |= hex_digits[ptr[1]] << 8;
    codepoint |= hex_digits[ptr[2]] << 4;
    codepoint |= hex_digits[ptr[3]];

    return codepoint;
}

static inline void parse_whitespace(const char** ptr, json_context_impl_t* context)
{
    (void)context;

    // most common cases : single space
    if (**ptr > ' ')
        return;
    if (**ptr == ' ' && *(*ptr + 1) > ' ')
    {
        ++*ptr;
        return;
    }
    if (**ptr == '\r')
        ++*ptr;

#ifdef __AVX2__
    // common too: \n and spaces
    {
        const __m256i data = _mm256_loadu_si256((const __m256i*)*ptr);
        const __m256i eol_spaces = _mm256_set_epi8(' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ','\n');

        const __m256i mask = _mm256_cmpeq_epi8(data, eol_spaces);
        const uint32_t movemask = ~_mm256_movemask_epi8(mask);
        if (movemask)
        {
            const int first_non_space = __builtin_ctz(movemask);
            *ptr += first_non_space;
        }
        else
        {
            *ptr += 32;
        }
    }

    if (**ptr > ' ')
        return;

    // try skipping regular spaces first
    __m256i data = _mm256_loadu_si256((const __m256i*)*ptr);
    __m256i mask = _mm256_cmpeq_epi8(data, _mm256_set1_epi8(' '));
    uint32_t movemask = ~_mm256_movemask_epi8(mask);
    if (movemask)
    {

        const int first_non_space = __builtin_ctz(movemask);
        *ptr += first_non_space;
        if (**ptr > ' ')
            return;

    }
    else
    {
        *ptr += 32;
    }

#endif

#if defined(__SSSE3__)

    const __m128i nrt_lut = _mm_set_epi8(0xFF, 0xFF, 0, 0xFF, 0xFF, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

    while (true)
    {

        const __m128i data = _mm_loadu_si128((const __m128i*)*ptr);
        const __m128i dong = _mm_min_epu8(data, _mm_set1_epi8(0x0F));
        const __m128i not_an_nrt_mask = _mm_shuffle_epi8(nrt_lut, dong);
        const __m128i space_mask = _mm_cmpeq_epi8(data, _mm_set1_epi8(' '));
        const __m128i non_whitespace_mask = _mm_xor_si128(not_an_nrt_mask, space_mask);
        const int movemask = _mm_movemask_epi8(non_whitespace_mask);
        if (__builtin_expect(movemask, 1))
        {
            const int first_non_whitespace = __builtin_ctz(movemask);
            *ptr += first_non_whitespace;
            return;

        }
        else
        {
            *ptr += 16;
            continue;
        }
    }

#elif defined(__SSE4_2__)

    const __m128i w = _mm_setr_epi8('\n','\r','\t', ' ', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    for (;; *ptr += 16) {
        const __m128i s = _mm_loadu_si128((const __m128i *)*ptr);
        const int r = _mm_cmpistri(w, s, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_LEAST_SIGNIFICANT | _SIDD_NEGATIVE_POLARITY);
        if (r != 16)    // some of characters is non-whitespace
        {
            *ptr += r;
            return;
        }
    }
#else
    while (**ptr == ' ' || **ptr == '\t' || **ptr == '\n'
           || **ptr == '\r')
        ++*ptr;
#endif
}

    static char escape_lut[256] = {
        [0 ... 255] = '\0',
        ['n'] = '\n',
        ['\\'] = '\\',
        ['/'] = '/',
        ['b'] = '\b',
        ['f'] = '\f',
        ['r'] = '\r',
        ['t'] = '\t',
        ['"'] = '"',
        };
static unsigned inline parse_string(unsigned int* str_idx, const char** ptr, json_context_impl_t* context)
{
    bool has_escaped = false;

    // check if we are in the trivial case of an unescaped <32bytes string
#ifdef __AVX2__
    {
        const __m256i data = _mm256_loadu_si256((const __m256i*)*ptr);
        const __m256i backslash_mask = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\\'));
        const __m256i quote_mask = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('"'));
        const uint32_t quote_movemask = _mm256_movemask_epi8(quote_mask);
        if (_mm256_testz_si256(backslash_mask, backslash_mask) && quote_movemask != 0)
        {
            const int len = __builtin_ctz(quote_movemask);
            const uint32_t idx_mask = (uint64_t)(0xFFFFFFFF) >> (32-len);

            const __m256i invalid_char_mask = _mm256_cmpeq_epi8(data, _mm256_min_epu8(data, _mm256_set1_epi8(0x1A))); // no unsigned 8-bit cmplt
            const uint32_t invalid_char_movemask = _mm256_movemask_epi8(invalid_char_mask);

            //JSON_ALIGN(string_buffer);
            *str_idx = context->string_buffer_brk;
            char* string = &context->string_buffer[context->string_buffer_brk];

            JSON_ALLOC(string_buffer, len+1);

            //_mm256_store_si256((__m256i*)string, data);
            memcpy(string, *ptr, 32);
            string[len] = '\0';
            *ptr += len+1;

            if ((invalid_char_movemask&idx_mask) != 0)
                error("control characters must be escaped", ptr, context);

            return len;
        }
    }
#endif

    *str_idx = context->string_buffer_brk;
    char* string = &context->string_buffer[context->string_buffer_brk];
    char* string_start = string;

    const uint8_t* start = (uint8_t*)*ptr;
    const uint8_t* end   = (uint8_t*)find_end_of_string_lit(*ptr, &has_escaped, context, string);
    const int len = end - start;


    // processed string can only be at most as long as the literal string
    // allocate 'len+1' chars on the string buffer

    JSON_ALLOC(string_buffer, len+1+32); // +1 for the null terminator, +32 to enable SSE/AVX copy optimizations

    if (!has_escaped)
    {
        // string has already been speculativery written by find_end_of_string_lit, adding the terminator byte is the only thing left to be done
        string[len] = '\0';

        *ptr = (const char*)(end + 1);

        return len; // return the length of the string
    }
    for (int i = 0; i < len; ++i)
    {
    loop:;
    // escape
#if defined (__AVX2__)
        const __m256i data = _mm256_loadu_si256((const __m256i*)&start[i]);
        const __m256i backslash_mask    = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\\'));
        const uint32_t movemask = _mm256_movemask_epi8(backslash_mask);
        if (movemask == 0)
        {
            _mm256_storeu_si256((__m256i*)string, data);
            string += 32;
            i += 32;
            if (i >= len)
                break;

            goto loop;
        }
        else
        {
            uint8_t count = __builtin_ctz(movemask);
            memcpy(string, &start[i], count);
            string += count;
            i += count;

            ++i;

            if (__builtin_expect(start[i] != 'u', 1))
            {
                uint8_t val = escape_lut[start[i]];
                if (__builtin_expect(val == '\0', 0))
                    error("invalid escape sequence", ptr, context);
                *string++ = val;
            }
            else
            {
                ++i;
                // expect 4 more digits
                if (!check_is_codepoint((uint8_t*)&start[i]))
                    error("invalid unicode escape", ptr, context);
                utf8_encode(unescape_codepoint((uint8_t*)&start[i]), &string);
            }
#elif defined(__SSE2__)
        const __m128i data = _mm_loadu_si128((const __m128i*)&start[i]);
        const __m128i backslash_mask    = _mm_cmpeq_epi8(data, _mm_set1_epi8('\\'));
        const uint16_t movemask = _mm_movemask_epi8(backslash_mask);
        if (movemask == 0)
        {
            const int copy_len = (i + 16 > len) ? len - i : 16;

            _mm_storeu_si128((__m128i*)string, data);
            string += copy_len;
            i += copy_len;
            if (i >= len)
                break;
            goto loop;
        }
        else
        {
            int count = __builtin_ctz(movemask);
            memcpy(string, &start[i], count);
            string += count;
            i += count;
            if (i >= len)
                break;
            goto loop;
        }
#else
        *string++ = value;
#endif
        }
    }
    *string = '\0';

    unsigned int actual_len = string - string_start;

    *ptr = (const char*)(end + 1);

    return actual_len; // return the length of the string
}


static void parse_element(json_value_t* val, const char** ptr, json_context_impl_t* context);

_Alignas(64) uint32_t char_table[(('9'+1) << 8) + ('9'+1)];

#if 0
static inline unsigned long long fast_atoi(const char** data)
{
    unsigned long long val = 0;
    do
    {
        val = val*10 + (**data&0xF);
        ++*data;
    } while(**data >= '0' && **data <= '9');
    return val;
}
#else

static inline uint64_t fast_atoi(const char** data)
{
    uint64_t val = 0;
    uint16_t value16 = **(uint16_t**)data;
    do
    {
        if (*(*data+1) >= '0' && *(*data+1) <= '9')
        {
            val = val*100 + char_table[value16];
            *data += 2;
            value16 = **(uint16_t**)data;
        }
        else
        {
            val = val*10 + value16&0xF;
            ++*data;
            break;
        }
    } while ((value16&0xFF) >= '0' && (value16&0xFF) <= '9');
    return val;
}
#endif

// double have up to 16 decimal places
static const double decimals[16][10] = {{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
                                        {0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09},
                                        {0.000, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009},
                                        {0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009},
                                        {0.00000, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009},
                                        {0.000000, 0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000007, 0.000008, 0.000009},
                                        {0.0000000, 0.0000001, 0.0000002, 0.0000003, 0.0000004, 0.0000005, 0.0000006, 0.0000007, 0.0000008, 0.0000009},
                                        {0.00000000, 0.00000001, 0.00000002, 0.00000003, 0.00000004, 0.00000005, 0.00000006, 0.00000007, 0.00000008, 0.00000009},
                                        {0.000000000, 0.000000001, 0.000000002, 0.000000003, 0.000000004, 0.000000005, 0.000000006, 0.000000007, 0.000000008, 0.000000009},
                                        {0.0000000000, 0.0000000001, 0.0000000002, 0.0000000003, 0.0000000004, 0.0000000005, 0.0000000006, 0.0000000007, 0.0000000008, 0.0000000009},
                                        {0.00000000000, 0.00000000001, 0.00000000002, 0.00000000003, 0.00000000004, 0.00000000005, 0.00000000006, 0.00000000007, 0.00000000008, 0.00000000009},
                                        {0.000000000000, 0.000000000001, 0.000000000002, 0.000000000003, 0.000000000004, 0.000000000005, 0.000000000006, 0.000000000007, 0.000000000008, 0.000000000009},
                                        {0.0000000000000, 0.0000000000001, 0.0000000000002, 0.0000000000003, 0.0000000000004, 0.0000000000005, 0.0000000000006, 0.0000000000007, 0.0000000000008, 0.0000000000009},
                                        {0.00000000000000, 0.00000000000001, 0.00000000000002, 0.00000000000003, 0.00000000000004, 0.00000000000005, 0.00000000000006, 0.00000000000007, 0.00000000000008, 0.00000000000009},
                                        {0.000000000000000, 0.000000000000001, 0.000000000000002, 0.000000000000003, 0.000000000000004, 0.000000000000005, 0.000000000000006, 0.000000000000007, 0.000000000000008, 0.000000000000009},
                                        {0.0000000000000000, 0.0000000000000001, 0.0000000000000002, 0.0000000000000003, 0.0000000000000004, 0.0000000000000005, 0.0000000000000006, 0.0000000000000007, 0.0000000000000008, 0.0000000000000009}
};

#define EXP10_TABLE_MAX 308
static bool _Atomic is_table_exp10_filled;
static double table_exp10[EXP10_TABLE_MAX*2+2];

static double fast_frac_atoi(const char** data)
{
    double val = 0.0;

    // un peu d'unrolling!
    val = decimals[0][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[1][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[2][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[3][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[4][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[5][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[6][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[7][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[8][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[9][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[10][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[11][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[12][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[13][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[14][**data&0xF]; ++*data;
    if (**data < '0' || **data > '9')
        return val;
    val += decimals[15][**data&0xF]; ++*data;
    // ignore remaining decimals as they are too small to matter with double-precision floats
    while (**data >= '0' && **data <= '9')
        ++*data;

    return val;
}

static bool parse_number(json_number_t* val, const char** ptr, json_context_impl_t* context)
{
    bool is_real = false;

    int sign = 1;

    if (accept(ptr, '-', context))
        sign = -1;

    long long int_part = 0;
    double frac_part = 0;
    unsigned long long u_exp_part = 0;
    long long exp_part = 0;

    if (**ptr == '0')
    {
        ++*ptr;
    }
    else
    {
        if (**ptr < '1' || **ptr > '9')
            error("invalid number", ptr, context);
        int_part = fast_atoi(ptr);
    }
    if (accept(ptr, '.', context))
    {
        is_real = true;

        if (**ptr < '0' || **ptr > '9')
            error("invalid number", ptr, context);
        frac_part = fast_frac_atoi(ptr);
    }
    // upper or lowercase ASCII e/E
    if ((**ptr|0x20) == 'e')
    {
        is_real = true;

        ++*ptr;
        bool exp_neg = 0;
        if (**ptr == '+')
            ++*ptr;
        else if  (**ptr == '-')
        {
            ++*ptr;
            exp_neg = true;
        }

        if (**ptr < '0' || **ptr > '9')
            error("invalid number", ptr, context);
        u_exp_part = fast_atoi(ptr);
        if (u_exp_part > EXP10_TABLE_MAX) // can happen in case of overflow
            u_exp_part = EXP10_TABLE_MAX;
        exp_part = u_exp_part;
        if (exp_neg)
            exp_part = -exp_part;
    }

    if (is_real)
    {
        val->num_real = (double)int_part + frac_part;
        val->num_real *= table_exp10[exp_part+EXP10_TABLE_MAX];
        val->num_real *= sign;
    }
    else
    {
        val->num_int = int_part*sign;
    }

    return is_real;
}

static bool compare4(const char* ptr, const char* str)
{
    if (ptr[0] == '\0' || ptr[1] == '\0' || ptr[2] == '\0')
        return false;
    return ptr[0] == str[0] && ptr[1] == str[1] && ptr[2] == str[2] && ptr[3] == str[3];
}

typedef struct json_marker_t
{
    json_value_t* base_ptr;
    int type; // 0 - brace; 1 - bracket
    unsigned int commas;
} json_marker_t;

#define JSON_STACK_SIZE 4096*16

#define JSON_PUSH_STACK() \
    ++sp_ptr; if (sp_ptr >= &json_stack[JSON_STACK_SIZE]) error("out of json stack memory", &ptr, context);
#define JSON_PUSH_BRACE() \
    marker_ptr->base_ptr = sp_ptr; marker_ptr->type = 0; marker_ptr->commas = 0; ++marker_ptr;  if (marker_ptr >= &json_markers[JSON_STACK_SIZE]) error("out of json stack memory", &ptr, context);
#define JSON_PUSH_BRACKET() \
    marker_ptr->base_ptr = sp_ptr; marker_ptr->type = 1; marker_ptr->commas = 0; ++marker_ptr;  if (marker_ptr >= &json_markers[JSON_STACK_SIZE]) error("out of json stack memory", &ptr, context);
#define JSON_POP_BRACKET() --marker_ptr; if (marker_ptr < &json_markers[0] || marker_ptr->type != 1) error("expected bracket", &ptr, context);
#define JSON_POP_BRACE() --marker_ptr; if (marker_ptr < &json_markers[0] || marker_ptr->type != 0) error("expected brace", &ptr, context);
#define JSON_REGISTER_COMMA() \
    ++(marker_ptr-1)->commas;
#define JSON_POP_STACK(x) --sp_ptr;

#ifndef JSON_STACK_WORKING_MEMORY
static json_value_t json_stack[JSON_STACK_SIZE];
static json_marker_t json_markers[JSON_STACK_SIZE];
#endif

static const char* parse_loop(const char* ptr, json_value_t* val_ptr, json_context_impl_t* context)
{
#ifdef JSON_STACK_WORKING_MEMORY
    json_value_t json_stack[JSON_STACK_SIZE];
    json_marker_t json_markers[JSON_STACK_SIZE];
#endif

    json_value_t* sp_ptr = &json_stack[1];
    json_marker_t* marker_ptr = &json_markers[1];

    parse_whitespace(&ptr, context);
    if (*ptr == '\0')
        error("no values in input", &ptr, context);

loop:
    parse_whitespace(&ptr, context);

no_ws_loop:

    switch (*ptr)
    {
        case '{':
            ++ptr;
            JSON_PUSH_BRACE();
            goto loop;
        case '[':
            ++ptr;
            if (*ptr == ']') // empty array, fast case
            {
                ++ptr;
                sp_ptr->type = JSON_ARRAY;
                sp_ptr->object.entry_count = 0;
                JSON_PUSH_STACK();
                goto loop;
            }
            JSON_PUSH_BRACKET();
            goto loop;
        case '}':
        {
            ++ptr;

            JSON_POP_BRACE();
            json_value_t* base_ptr = marker_ptr->base_ptr;
            unsigned int count = sp_ptr - base_ptr;

            if (count == 0)
            {
                if (marker_ptr->commas != 0)
                    error("invalid number of commas", &ptr, context);
            }
            else if (count/2-1 != marker_ptr->commas) // halved because of key/value pairs
                error("invalid number of commas", &ptr, context);

            JSON_ALLOC(key_val_buffer, count/2);

            const unsigned start_idx = context->key_val_buffer_brk - count/2;

            if (count%2) // not key/value pairs only if count isn't even
                error("object doesn't contain key/value pairs only", &ptr, context);
            for (unsigned i = 0; i < count/2; ++i)
            {
                if (base_ptr[i*2].type != JSON_KEY)
                    error("expected a key", &ptr, context);
                if (base_ptr[i*2+1].type == JSON_KEY)
                    error("expected a value", &ptr, context);
                context->key_val_buffer[start_idx+i].str_idx = base_ptr[i*2].str_idx;
                context->key_val_buffer[start_idx+i].str_len = base_ptr[i*2].str_len;
                context->key_val_buffer[start_idx+i].value   = base_ptr[i*2+1];
                context->key_val_buffer[start_idx+i].next_value_idx = start_idx+i+1;
            }

            sp_ptr -= count;

            sp_ptr->type = JSON_OBJECT;
            sp_ptr->object.start_idx = start_idx;
            sp_ptr->object.entry_count = count/2;
            JSON_PUSH_STACK();

            if (*ptr == ',')
            {
                JSON_REGISTER_COMMA();
                ++ptr;
            }

            goto loop;
        }
        case ']': // pop all values off the stack until the last '['
        {
            ++ptr;

            JSON_POP_BRACKET();
            json_value_t* base_ptr = marker_ptr->base_ptr;
            unsigned int count = sp_ptr - base_ptr;

            unsigned int count_m_one = (count == 0) ? 0 : count-1;
            if (count_m_one != marker_ptr->commas)
                error("invalid number of commas", &ptr, context);

            JSON_ALLOC(key_val_buffer, count);

            const unsigned start_idx = context->key_val_buffer_brk - count;

            for (unsigned i = 0; i < count; ++i)
            {
                context->key_val_buffer[start_idx+i].value = base_ptr[i];
                context->key_val_buffer[start_idx+i].next_value_idx = i+1;
            }

            sp_ptr -= count;

            sp_ptr->type = JSON_ARRAY;
            sp_ptr->array.start_idx = start_idx;
            sp_ptr->array.entry_count = count;

            JSON_PUSH_STACK();

            if (*ptr == ',')
            {
                JSON_REGISTER_COMMA();
                ++ptr;
            }

            goto loop;
        }
        case ':':
            ++ptr;
            if ((sp_ptr-1)->type != JSON_STRING)
                error("key isn't a string", &ptr, context);
            (sp_ptr-1)->type = JSON_KEY;
            goto loop;
        case ',':
            ++ptr;
            JSON_REGISTER_COMMA();
            parse_whitespace(&ptr, context);
            if (*ptr != '"')
                goto loop;
            else
                ; // falthrough !
        case '"':
        {
            ++ptr;
            sp_ptr->str_len = parse_string(&sp_ptr->str_idx, &ptr, context);

            if (*ptr == ' ')
                ++ptr;
            // optimization as it is common
            if (*ptr == ':' && ptr[1] == ' ' && ptr[2] > ' ')
            {
                ptr += 2;
                sp_ptr->type = JSON_KEY;

                JSON_PUSH_STACK();

                goto no_ws_loop;
            }

            sp_ptr->type = JSON_STRING;

            JSON_PUSH_STACK();

            if (*ptr == ',')
            {
                JSON_REGISTER_COMMA();
                ++ptr;
            }

            goto loop;
        }
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
        case '-':
        {
            json_number_t num;
            bool is_real = parse_number(&num, &ptr, context);
            sp_ptr->num = num;
            sp_ptr->type = is_real ? JSON_NUMBER_FLOAT : JSON_NUMBER_INT;
            JSON_PUSH_STACK();


            if (*ptr == ',')
            {
                JSON_REGISTER_COMMA();
                ++ptr;
            }
            goto loop;
        }
        case 't':
        {
            if (compare4(ptr, "true") == 0)
                error("invalid json value", &ptr, context);

            sp_ptr->type = JSON_BOOL;
            sp_ptr->boolean = true;
            JSON_PUSH_STACK();
            ptr += 4;

            if (*ptr == ',')
            {
                JSON_REGISTER_COMMA();
                ++ptr;
            }

            goto loop;
        }
        case 'f':
        {
            if (compare4((ptr+1), "alse") == 0)
                error("invalid json value", &ptr, context);

            sp_ptr->type = JSON_BOOL;
            sp_ptr->boolean = false;
            JSON_PUSH_STACK();
            ptr += 5;

            if (*ptr == ',')
            {
                JSON_REGISTER_COMMA();
                ++ptr;
            }

            goto loop;
        }
        case 'n':
        {
            if (compare4(ptr, "null") == 0)
                error("invalid json value", &ptr, context);

            sp_ptr->type = JSON_NULL;
            JSON_PUSH_STACK();
            ptr += 4;

            if (*ptr == ',')
            {
                JSON_REGISTER_COMMA();
                ++ptr;
            }

            goto loop;
        }
        case '\0':
            *val_ptr = json_stack[1];
            goto out;
        default:
            error("invalid json value", &ptr, context);
    }

out:
    if (marker_ptr != &json_markers[1])
        error("unmatched brackets/braces", &ptr, context);
    if (json_markers[0].commas != 0)
        error("extra commas found", &ptr, context);
    if (sp_ptr != &json_stack[1+1])
        error("unexpected extra values", &ptr, context);
    return ptr;
}

static void init_table_exp10()
{
    for (int i = -EXP10_TABLE_MAX; i <= EXP10_TABLE_MAX; ++i)
        table_exp10[i+EXP10_TABLE_MAX] = pow(10, i);

    is_table_exp10_filled = true;
}

json_result_t parse_json(const char *input, int input_size,
                         json_context_t *context)
{
    if (!is_table_exp10_filled)
    {
        for (size_t i = 0; i < 10; ++i)
        {
            for (size_t j = 0; j < 10; ++j)
            {
                char_table[(i+'0')<<8 | (j+'0')] = j*10 + i;
            }
        }
        init_table_exp10();
    }

    // consume the utf-8 BOM if it is present
    if (strncmp(input, "\xEF\xBB\xBF", 3) == 0)
    {
        input += 3;
        input_size -= 3;
    }

    json_context_impl_t context_impl;
    context_impl.source = input;
    context_impl.nest_depth = 0;
    context_impl.string_buffer = context->string_buffer;
    context_impl.string_buffer_size = context->string_buffer_size;
    context_impl.string_buffer_brk = 0;
    context_impl.string_buffer_realloc = context->string_realloc;
    context_impl.key_val_buffer = context->key_val_buffer;
    context_impl.key_val_buffer_size = context->key_val_buffer_size;
    context_impl.key_val_buffer_brk = 0;
    context_impl.key_val_buffer_realloc = context->key_val_realloc;

    json_value_t root;
    root.type = JSON_NULL;
    if (setjmp(context_impl.error_jmp_buf) == 0)
    {
        parse_whitespace(&input, &context_impl);
        input = parse_loop(input, &root, &context_impl);
#ifndef JSON_IGNORE_TRAILING_GARBAGE
        parse_whitespace(&input, &context_impl);
        if (input < context_impl.source + input_size) // extra garbage
        {
            context_impl.result.accepted = false;
            context_impl.result.error.reason = "trailing garbage";
        }
        else
#endif
        {
            context_impl.result.accepted = true;
            context_impl.result.value = root;
        }

        return context_impl.result;
    }
    else // error path
    {
        context_impl.result.accepted = false;
        return context_impl.result;
    }
}

static uint32_t digit_table[10000];
static bool _Atomic is_digit_table_filled;

static void fill_digit_table()
{
    int table_counter = 0;
    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            for (int k = 0; k < 10; ++k)
            {
                for (int l = 0; l < 10; ++l)
                {
                    uint32_t d1 = l + '0';
                    uint32_t d2 = k + '0';
                    uint32_t d3 = j + '0';
                    uint32_t d4 = i + '0';
                    uint32_t val = d1 | (d2 << 8) | (d3 << 16) | (d4 << 24);
                    digit_table[table_counter++] = val;
                }
            }
        }
    }
}


static char* print_integer(int64_t val, char* ptr)
{
    // makes the following code simpler
    if (val == 0)
    {
        *ptr++ = '0';
        return ptr;
    }

    char reverse_buffer[20];
    unsigned int reverse_buffer_idx = 0;
    memset(reverse_buffer, '0', 20);

    if (val < 0)
    {
        val = -val;
        *ptr++ = '-';
    }

    do
    {
        uint64_t digit = val % 10000;
        val /= 10000;
        memcpy(&reverse_buffer[reverse_buffer_idx], &digit_table[digit], 4);
        reverse_buffer_idx+=4;
    } while (val != 0);

    // ignore leading zeroes
    while (reverse_buffer[reverse_buffer_idx-1] == '0')
        --reverse_buffer_idx;

    while (reverse_buffer_idx > 0)
    {
        --reverse_buffer_idx;
        *ptr++ = reverse_buffer[reverse_buffer_idx];
    }
    return ptr;
}

static char* print_double(double val, char* ptr)
{
#ifdef JSON_FAST_DOUBLE_PRINT
    char frac_part_buffer[40];
    memset(frac_part_buffer, '0', 20);

    if (val<0)
    {
        val = -val;
        *ptr++ = '-';
    }

    uint64_t int_val = (uint64_t)val;
    ptr = print_integer(int_val, ptr);
    *ptr++ = '.';

    double frac = val - int_val;
    uint64_t frac_int = (uint64_t)(frac*JSON_FAST_DOUBLE_PRINT_PRECISION_EXP); // 8 decimal places

    const char* frac_part_buffer_end = print_integer(frac_int, frac_part_buffer+20);
    int frac_part_len = frac_part_buffer_end - (frac_part_buffer+20);
    int shift = JSON_FAST_DOUBLE_PRINT_PRECISION - frac_part_len;
    const char* actual_fract_part_start = frac_part_buffer+20 - shift;
    for (int i = 0; i < frac_part_len; ++i)
        *ptr++ = actual_fract_part_start[i];
#else
    extern int fpconv_dtoa(double d, char dest[24]);
    ptr += fpconv_dtoa(val, ptr);
#endif
    return ptr;
}

#define PRINTC(c) \
    **ptr = c; ++*ptr; if (*ptr >= end_of_buf) return;
#define PRINTSTR(str, n) \
    if (*ptr+n >= end_of_buf) return; \
    memcpy(*ptr, str, n); *ptr += n;
#define PRINTNEWLINE() \
    if (*ptr + depth+1 >= end_of_buf) return; \
    **ptr = '\n'; ++*ptr; \
    for (unsigned int i = 0; i < depth; ++i) \
    { **ptr = ' '; ++*ptr; }


#ifndef __SSE4_2__
static void print_json_str(json_context_t *info, unsigned int str_key, unsigned int str_len, char** ptr, const char* end_of_buf)
{
    // take a generous margin
    if (*ptr + str_len*2+2 >= end_of_buf)
        return;

    const char* key = &info->string_buffer[str_key];
    char* ptr_copy = *ptr;

    *ptr_copy++ = '"';
    for (unsigned int i = 0; i < str_len; ++i)
    {
        switch (key[i])
        {
            case '\n':
                *ptr_copy++ = '\\'; *ptr_copy++ = 'n'; break;
            case '\\':
                *ptr_copy++ = '\\'; *ptr_copy++ = '\\'; break;
            case '\b':
                *ptr_copy++ = '\\'; *ptr_copy++ = 'b'; break;
            case '\f':
                *ptr_copy++ = '\\'; *ptr_copy++ = 'f'; break;
            case '\r':
                *ptr_copy++ = '\\'; *ptr_copy++ = 'r'; break;
            case '\t':
                *ptr_copy++ = '\\'; *ptr_copy++ = 't'; break;
            case '"':
                *ptr_copy++ = '\\'; *ptr_copy++ = '"'; break;
            default:
                *ptr_copy++ = key[i];
        }
    }
    *ptr_copy++ = '"';

    *ptr = ptr_copy;
}
#else
static _Alignas(16) const char escape_chars_vector[16] = "\n\\\b\f\t\"";
static void print_json_str(json_context_t *info, unsigned int str_key, unsigned int str_len, char** ptr, const char* end_of_buf)
{
    // take a generous margin
    if (*ptr + str_len*2+2 >= end_of_buf)
        return;

    const __m128i e = _mm_load_si128((const __m128i *)escape_chars_vector);

    const char* key = &info->string_buffer[str_key];
    char* ptr_copy = *ptr;

    *ptr_copy++ = '"';
    for (unsigned int i = 0; i < str_len;)
    {
        const int len = (i + 16 > str_len) ? str_len - i : 16;

        const __m128i s = _mm_loadu_si128((const __m128i *)&key[i]);
        const int r = _mm_cmpestrc(e, 6, s, len, _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK | _SIDD_POSITIVE_POLARITY);
        if (r) // contains escaped characters
        {
            for (int j = 0; j < 16 && i < str_len; ++i, ++j)
            {
                switch (key[i])
                {
                    case '\n':
                        *ptr_copy++ = '\\'; *ptr_copy++ = 'n'; break;
                    case '\\':
                        *ptr_copy++ = '\\'; *ptr_copy++ = '\\'; break;
                    case '\b':
                        *ptr_copy++ = '\\'; *ptr_copy++ = 'b'; break;
                    case '\f':
                        *ptr_copy++ = '\\'; *ptr_copy++ = 'f'; break;
                    case '\r':
                        *ptr_copy++ = '\\'; *ptr_copy++ = 'r'; break;
                    case '\t':
                        *ptr_copy++ = '\\'; *ptr_copy++ = 't'; break;
                    case '"':
                        *ptr_copy++ = '\\'; *ptr_copy++ = '"'; break;
                    default:
                        *ptr_copy++ = key[i];
                }
            }
        }
        else
        {
            // fast path
            _mm_storeu_si128((__m128i*)ptr_copy, s);
            ptr_copy += len;
            i += len;
        }
    }

    *ptr_copy++ = '"';

    *ptr = ptr_copy;
}
#endif

static void print_json_value(json_context_t *info, json_value_t* value, char** ptr, const char* end_of_buf, unsigned int depth, bool pretty)
{
    switch (value->type)
    {
        case JSON_OBJECT:
        {
            if (pretty)
            {
                PRINTNEWLINE(); PRINTC('{'); ++depth; PRINTNEWLINE();
            }
            else
                PRINTC('{');

            unsigned int current_idx = value->object.start_idx;
            for (unsigned int i = 0; i < value->object.entry_count; ++i)
            {
                json_key_value_t key_val = info->key_val_buffer[current_idx];
                print_json_str(info, key_val.str_idx, key_val.str_len, ptr, end_of_buf);
                PRINTSTR(" : ", 3);
                print_json_value(info, &key_val.value, ptr, end_of_buf, depth+1, pretty);
                if (i != value->object.entry_count-1)
                {
                    PRINTC(',');
                    if (pretty)
                        PRINTNEWLINE();
                }

                current_idx = key_val.next_value_idx;
            }
            if (pretty)
            {
                --depth; PRINTNEWLINE(); PRINTC('}');
            }
            else
                PRINTC('}');
        }
        break;
        case JSON_ARRAY:
        {
            PRINTC('[');
            unsigned int current_idx = value->array.start_idx;
            for (unsigned int i = 0; i < value->array.entry_count; ++i)
            {
                json_key_value_t key_val = info->key_val_buffer[current_idx];
                print_json_value(info, &key_val.value, ptr, end_of_buf, depth+1, pretty);
                if (i != value->object.entry_count-1)
                {
                    PRINTSTR(", ", 2);
                }

                current_idx = key_val.next_value_idx;
            }
            PRINTC(']');
        }
        break;
        case JSON_STRING:
            print_json_str(info, value->str_idx, value->str_len, ptr, end_of_buf);
            break;
        case JSON_NUMBER_INT:
        case JSON_NUMBER_FLOAT:
        {
            // not enough space available
            if (*ptr + DBL_MAX_10_EXP >= end_of_buf) return;
            if (value->type == JSON_NUMBER_FLOAT)
            {
                *ptr = print_double(value->num.num_real, *ptr);
            }
            else
            {
                *ptr = print_integer(value->num.num_int, *ptr);
            }
        }
        break;
        case JSON_BOOL:
        {
            if (value->boolean == true)
            {
                PRINTSTR("true", 4);
            }
            else
            {
                PRINTSTR("false", 5);
            }
        }
        break;
        case JSON_NULL:
            PRINTSTR("null", 4);
            break;
    }
}

unsigned int print_json(json_value_t *value, json_context_t *context, bool pretty, char *buffer, unsigned int buffer_size)
{
    if (!is_digit_table_filled)
    {
        fill_digit_table();
        is_digit_table_filled = true;
    }

    if (!value)
        return 0;

    const char* buf_start  = buffer;
    const char* end_of_buf = buffer + buffer_size;
    print_json_value(context, value, &buffer, end_of_buf, 0, pretty);
    *buffer = '\0';

    unsigned int len = buffer - buf_start;

    return len;
}

static inline bool is_string_digits(const char* ptr)
{
    while (*ptr)
    {
        if (*ptr < '0' || *ptr > '9')
            return false;
        ++ptr;
    }
    return true;
}

// https://tools.ietf.org/html/rfc6901
json_value_t *json_pointer(const char *pointer, json_value_t *root, json_context_t *context)
{
    char key_buffer[256];
    char key_buffer_unescaped[256];
    unsigned int key_buffer_idx = 0;

    while (*pointer++ == '/')
    {
        key_buffer_idx = 0;

        while (*pointer && *pointer != '/')
        {
            if (key_buffer_idx < 256-1)
            {
                key_buffer_unescaped[key_buffer_idx] = *pointer;
                ++key_buffer_idx;
            }
            ++pointer;
        }
        key_buffer_unescaped[key_buffer_idx] = '\0';

        // escape control characters in the key_buffer
        char* key_buf_ptr = key_buffer;
        for (unsigned i = 0; i < key_buffer_idx; ++i)
        {
            if (i != key_buffer_idx-1 && key_buffer_unescaped[i] == '~')
            {
                *key_buf_ptr++ = key_buffer_unescaped[i] == '0' ? '~' : '/';
                ++i;
            }
            else
                *key_buf_ptr++ = key_buffer_unescaped[i];
        }
        if (root->type == JSON_OBJECT)
        {
            unsigned int current_idx = root->object.start_idx;
            for (unsigned i = 0; i < root->object.entry_count; ++i)
            {
                json_key_value_t* key_val = &context->key_val_buffer[current_idx];
                // found!
                if (key_val->str_len == key_buffer_idx
                    && memcmp(key_buffer, &context->string_buffer[key_val->str_idx], key_val->str_len) == 0)
                {
                    root = &key_val->value;
                    goto found;
                }

                current_idx = key_val->next_value_idx;
            }
            return NULL;
        found:
            ;
        }
        else if (root->type == JSON_ARRAY)
        {
            const char** key_ptr = (const char**)&key_buffer;
            if (key_buffer_idx >= 20)
                return NULL; // index is wayyyy too long to be an actual index
            if (!is_string_digits(key_buffer))
                return NULL; // not an index either
            unsigned int idx = fast_atoi(key_ptr);
            if (idx >= root->array.entry_count)
                return NULL;
            unsigned int current_idx = root->object.start_idx;
            for (unsigned i = 0; i < idx; ++i)
            {
                json_key_value_t* key_val = &context->key_val_buffer[current_idx];
                current_idx = key_val->next_value_idx;
            }
            root = &context->key_val_buffer[current_idx].value;
        }
        else
            return NULL;

    }

    return root;
}

static json_key_value_t* json_create_keyval(json_context_t *context)
{
    if (context->key_val_buffer_brk + 1 >= context->key_val_buffer_size) {
        if (!context->key_val_realloc)
            return NULL;
        context->key_val_buffer = context->key_val_realloc(context->key_val_buffer, context->key_val_buffer_brk + 1);
        if (!context->key_val_buffer)
            return NULL;
    }
    context->key_val_buffer_brk += 1;
    return &context->key_val_buffer[context->key_val_buffer_brk-1];
}

json_value_t *json_create_value(json_context_t *context, uint8_t type)
{
    json_key_value_t* key_val = json_create_keyval(context);
    if (!key_val)
        return NULL;
    memset(key_val, 0, sizeof(json_key_value_t));
    key_val->value.type = type;
    return &key_val->value;
}

static void create_string(json_context_t* context, const char* key, unsigned int* str_key, unsigned int* str_len)
{
    *str_len = strlen(key);

    if (context->key_val_buffer_brk + *str_len >= context->key_val_buffer_size) {
        if (!context->key_val_realloc)
            return;
        context->key_val_buffer = context->key_val_realloc(context->key_val_buffer, context->key_val_buffer_brk + 1);
        if (!context->key_val_buffer)
            return;
    }
    *str_key = context->key_val_buffer_brk;
    memcpy(&context->string_buffer[context->key_val_buffer_brk], key, *str_len);
    context->key_val_buffer_brk += *str_len;
}

void json_object_add(json_context_t* context, json_value_t *object, const char *key, json_value_t *value)
{
    if (object->type != JSON_OBJECT)
        return;

    unsigned int* tail_idx = &object->object.start_idx;
    for (unsigned int i = 0; i < value->object.entry_count; ++i)
    {
        json_key_value_t* key_val = &context->key_val_buffer[*tail_idx];
        tail_idx = &key_val->next_value_idx;
    }
    json_key_value_t* key_val = json_create_keyval(context);
    if (!key_val)
        return;
    unsigned int value_idx = context->key_val_buffer_brk-1;
    *tail_idx = value_idx;
    ++object->object.entry_count;

    create_string(context, key, &key_val->str_idx, &key_val->str_len);
    key_val->value = *value;
}

void json_array_add(json_context_t* context, json_value_t *object, json_value_t *value)
{
    if (object->type != JSON_ARRAY)
        return;

    unsigned int* tail_idx = &object->array.start_idx;
    for (unsigned int i = 0; i < value->array.entry_count; ++i)
    {
        json_key_value_t* key_val = &context->key_val_buffer[*tail_idx];
        tail_idx = &key_val->next_value_idx;
    }
    json_key_value_t* key_val = json_create_keyval(context);
    if (!key_val)
        return;
    unsigned int value_idx = context->key_val_buffer_brk-1;
    *tail_idx = value_idx;
    ++object->array.entry_count;

    key_val->value = *value;
}

const char *json_get_string(json_value_t *val, json_context_t *context)
{
    if (val->type != JSON_STRING)
        return "";

    return &context->string_buffer[val->str_idx];
}

json_value_t *json_create_string(const char *str, json_context_t *context)
{
    json_value_t* val = json_create_value(context, JSON_STRING);
    create_string(context, str, &val->str_idx, &val->str_len);

    return val;
}
