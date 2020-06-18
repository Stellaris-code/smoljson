#include "json.h"

#include <string.h>
#include <stdint.h>
#include <setjmp.h>
#include <math.h>

#ifdef __SSE4_2__
#include <nmmintrin.h>
#endif
#ifdef __SSE2__
#include <pmmintrin.h>
#endif
#ifdef __AVX2__
#include <immintrin.h>
#endif

//#define JSON_IGNORE_TRAILING_GARBAGE
#define JSON_NEST_LIMIT 1024

#define JSON_ALLOC(buf_name, new_len) \
    if (context->buf_name##_brk + new_len >= context->buf_name##_size) { \
        if (!context->buf_name##_realloc) \
            error("out of " #buf_name " memory", ptr, context); \
        context->buf_name = context->buf_name##_realloc(context->buf_name, context->buf_name##_brk + new_len); \
        if (!context->buf_name) \
            error("out of " #buf_name " memory", ptr, context);} \
    context->buf_name##_brk += new_len;

typedef struct json_context_t
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
} json_context_t;

_Noreturn static inline void error(const char* reason, const char** ptr, json_context_t* context)
{
    context->result.error.reason = reason;
    context->result.error.index = *ptr - context->source;
    longjmp(context->error_jmp_buf, 1);
}

static inline void expect(const char** ptr, char c, json_context_t* context)
{
    if (**ptr != c)
    {
        error("unexpected token found", ptr, context);
    }
    else
    {
        ++*ptr;
    }
}
static inline bool accept(const char** ptr, char c, json_context_t* context)
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
    int backslash_count = 0;
    while (ptr[-backslash_count-1] == '\\')
        ++backslash_count;
    return backslash_count&1;
}

static inline const char* find_end_of_string_lit(const char* str, bool* has_escaped_characters, json_context_t* context)
{
#if defined(__AVX2__)
    *has_escaped_characters = false;

    for(;;str += 32)
    {
        const __m256i data = _mm256_loadu_si256((const __m256i*)str);
        const __m256i backslash_mask    = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\\'));
        const __m256i quote_mask        = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('"'));
        const __m256i invalid_char_mask = _mm256_cmpeq_epi8(data, _mm256_min_epu8(data, _mm256_set1_epi8(0x1A))); // no unsigned 8-bit cmplt
        uint32_t quote_movemask = _mm256_movemask_epi8(quote_mask);
        const uint32_t invalid_char_movemask = _mm256_movemask_epi8(invalid_char_mask);
        const uint32_t backslash_movemask = _mm256_movemask_epi8(backslash_mask);
        if (quote_movemask)
        {
            const uint32_t first_idx = __builtin_ctz(quote_movemask);
            // check that it's an actual quote, not an escaped quote
            if (str[-1] == '\\' || backslash_movemask != 0)
            {
                // iterate over every possible index
                uint32_t idx = 0;
                while (quote_movemask != 0)
                {
                    while ((quote_movemask&1) == 0)
                    {
                        ++idx;
                        quote_movemask >>= 1;
                    }
                    if (!is_quote_escaped(&str[idx])) // even amount of backslashes <=> unescaped quote
                    {
                        const uint32_t idx_mask = (uint64_t)(0xFFFFFFFF) >> (32-idx);

                        if ((invalid_char_movemask&idx_mask) != 0)
                            error("control characters must be escaped", &str, context);
                        *has_escaped_characters |= backslash_movemask&idx_mask;

                        str += idx;
                        return str;
                    }
                    quote_movemask >>= 1; ++idx;
                }
            }
            // no need to check, there aren't any backslashes in this chunk nor in the previous character
            else
            {
                const uint32_t idx_mask = (uint64_t)(0xFFFFFFFF) >> (32-first_idx);

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
            *has_escaped_characters |= backslash_movemask;
        }
    }
#elif defined(__SSE2__)
    *has_escaped_characters = false;

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
                int idx = 0;
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
                        *has_escaped_characters |= backslash_movemask&idx_mask;

                        str += idx;
                        return str;
                    }
                    quote_movemask >>= 1; ++idx;
                }
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
            *has_escaped_characters |= backslash_movemask;
        }
    }
#else
    const char* str_start = str;
    *has_escaped_characters = false;
    while (*str)
    {
        if (str[0] == '\\')
        {
            *has_escaped_characters = true;
        }
        else if (*(uint8_t*)str < 0x20) // unescaped control characted
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

static _Alignas(16) const char whitespace_vector[16] = " \n\r\t";
static inline void parse_whitespace(const char** ptr, json_context_t* context)
{
    (void)context;
    // fast path
    if (**ptr == ' ' || **ptr == '\n' || **ptr == '\r' || **ptr == '\t')
        ++*ptr;
    else
        return;

    if (**ptr > ' ')
        return;

#if defined(__AVX2__)&&defined(AVX2_WHITESPACE) // overhead is too significant

    for (;; *ptr += 32) {
        const __m256i data = _mm256_loadu_si256((const __m256i *)*ptr);
        const __m256i s = _mm256_cmpeq_epi8(data,  _mm256_set1_epi8(' '));
        const __m256i r = _mm256_cmpeq_epi8(data,  _mm256_set1_epi8('\r'));
        const __m256i t = _mm256_cmpeq_epi8(data,  _mm256_set1_epi8('\t'));
        const __m256i n = _mm256_cmpeq_epi8(data,  _mm256_set1_epi8('\n'));
        const __m256i mask = _mm256_or_si256(_mm256_or_si256(s, r), _mm256_or_si256(t, n));
        const uint32_t movemask = ~_mm256_movemask_epi8(mask);

        if (movemask != 0)    // some of characters is non-whitespace
        {
            const int idx = __builtin_ctz(movemask);
            *ptr += idx;
            return;
        }
    }
#elif defined(__SSE4_2__)

    const __m128i w = _mm_load_si128((const __m128i *)whitespace_vector);

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

static unsigned parse_string(char** str, const char** ptr, json_context_t* context)
{
    ++*ptr; // skip the first '"'

    bool has_escaped_characters;
    const uint8_t* start = (uint8_t*)*ptr;
    const uint8_t* end   = (uint8_t*)find_end_of_string_lit(*ptr, &has_escaped_characters, context);
    const int len = end - start;

    // processed string can only be at most as long as the literal string
    // allocate 'len+1' chars on the string buffer

    char* string = &context->string_buffer[context->string_buffer_brk];
    char* string_start = string;

    JSON_ALLOC(string_buffer, len+1+16); // +1 for the null terminator, +16 to enable SSE copy optimizations

    if (!has_escaped_characters)
    {
        // we can simply do a memcpy
        memcpy(string_start, start, len);
    }
    else
    {
        for (int i = 0; i < len; ++i)
        {
            uint8_t value = start[i];
            // escape
            if (value == '\\')
            {
                ++i;
                if (i == len)
                    error("invalid escape sequence", ptr, context);

                switch (start[i])
                {
                    case 'n':
                        *string++ = '\n';
                        break;
                    case '\\':
                        *string++ = '\\';
                        break;
                    case '/':
                        *string++ = '/';
                        break;
                    case 'b':
                        *string++ = '\b';
                        break;
                    case 'f':
                        *string++ = '\f';
                        break;
                    case 'r':
                        *string++ = '\r';
                        break;
                    case 't':
                        *string++ = '\t';
                        break;
                    case 'u':
                        ++i;
                        // expect 4 more digits
                        if (!check_is_codepoint((uint8_t*)&start[i]))
                            error("invalid unicode escape", ptr, context);
                        utf8_encode(unescape_codepoint((uint8_t*)&start[i]), &string);
                        break;
                    case '"':
                        *string++ = '"';
                        break;
                    default:
                        error("unknown escape sequence", ptr, context);
                }
            }
            else
            {
#if defined(__AVX2__)&&defined(AVX2_STRING)
                const __m256i data = _mm256_loadu_si256((const __m256i*)&start[i]);
                const __m256i backslash_mask    = _mm256_cmpeq_epi8(data, _mm256_set1_epi8('\\'));
                if (_mm256_testz_si256(backslash_mask,backslash_mask))
                {
                    _mm256_storeu_si256((__m256i*)string, data);
                    string += 32;
                    i += 31;
                }
                else
                    *string++ = value;
#elif defined(__SSE2__)
                const __m128i data = _mm_loadu_si128((const __m128i*)&start[i]);
                const __m128i backslash_mask    = _mm_cmpeq_epi8(data, _mm_set1_epi8('\\'));
                if (_mm_test_all_zeros(backslash_mask,backslash_mask))
                {
                    _mm_storeu_si128((__m128i*)string, data);
                    string += 16;
                    i += 15;
                }
                else
                    *string++ = value;
#else
                *string++ = value;
#endif
            }
        }
    }
    *string = '\0';

    *ptr = (const char*)(end + 1);

    *str = string_start;

    return string - string_start; // return the lenght of the string
}


static void parse_element(json_value_t* val, const char** ptr, json_context_t* context);
static void parse_object(json_object_t* object, const char** ptr, json_context_t* context)
{
    ++*ptr; // skip the '{'
    ++context->nest_depth;

    if (context->nest_depth > JSON_NEST_LIMIT)
        error("depth limit exceeded", ptr, context);

    object->entry_count = 0;
    object->start_idx = context->key_val_buffer_brk;

    parse_whitespace(ptr, context);
    if (**ptr != '}') // not an empty object
    {
        do
        {
            parse_whitespace(ptr, context);
            char* key;
            parse_string(&key, ptr, context);
            parse_whitespace(ptr, context);

            expect(ptr, ':', context);

            JSON_ALLOC(key_val_buffer, 1);
            parse_element(&context->key_val_buffer[context->key_val_buffer_brk-1].value, ptr, context);
        } while (accept(ptr, ',', context));
    }

    object->entry_count = context->key_val_buffer_brk - object->start_idx;

    expect(ptr, '}', context);

    parse_whitespace(ptr, context);

    --context->nest_depth;
}

static void parse_array(json_array_t* array, const char** ptr, json_context_t* context)
{
    ++*ptr; // skip the '['
    ++context->nest_depth;

    if (context->nest_depth > JSON_NEST_LIMIT)
        error("depth limit exceeded", ptr, context);

    array->entry_count = 0;
    array->start_idx = context->key_val_buffer_brk;

    parse_whitespace(ptr, context);
    if (**ptr != ']')
    {
        do
        {
            JSON_ALLOC(key_val_buffer, 1);
            parse_element(&context->key_val_buffer[context->key_val_buffer_brk-1].value, ptr, context);
        } while (accept(ptr, ',', context));
    }

    array->entry_count = context->key_val_buffer_brk - array->start_idx;

    expect(ptr, ']', context);

    parse_whitespace(ptr, context);

    --context->nest_depth;
}

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

static bool parse_number(json_number_t* val, const char** ptr, json_context_t* context)
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

static void parse_element(json_value_t *val, const char** ptr, json_context_t* context)
{
    parse_whitespace(ptr, context);

    switch (**ptr)
    {
        case '{':
            val->type = JSON_OBJECT;
            parse_object(&val->object, ptr, context);
            return;
        case '[':
            val->type = JSON_ARRAY;
            parse_array(&val->array, ptr, context);
            return;
        case '"':
            val->type = JSON_STRING;
            val->str_len = parse_string((char**)&val->string, ptr, context);
            parse_whitespace(ptr, context);
            return;
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
            val->type = JSON_NUMBER;
            val->is_real = parse_number(&val->num, ptr, context);
            parse_whitespace(ptr, context);
            return;
        case 't':
            if (compare4(*ptr, "true") == 0)
                error("invalid json value", ptr, context);
            val->type = JSON_BOOL;
            val->boolean = true;
            *ptr += 4;
            parse_whitespace(ptr, context);
            return;
        case 'f':
            if (compare4((*ptr+1), "alse") == 0)
                error("invalid json value", ptr, context);
            val->type = JSON_BOOL;
            val->boolean = false;
            *ptr += 5;
            parse_whitespace(ptr, context);
            return;
        case 'n':
            if (compare4(*ptr, "null") == 0)
                error("invalid json value", ptr, context);
            val->type = JSON_NULL;
            *ptr += 4;
            parse_whitespace(ptr, context);
            return;
        default:
            error("invalid json value", ptr, context);
    }
}

static void init_table_exp10()
{
    for (int i = -EXP10_TABLE_MAX; i <= EXP10_TABLE_MAX; ++i)
        table_exp10[i+EXP10_TABLE_MAX] = pow(10, i);

    is_table_exp10_filled = true;
}

json_result_t parse_json(const char *input, int input_size,
                         char* string_buffer, unsigned int string_buffer_size, realloc_callback_t* string_realloc,
                         json_key_value_t* key_val_buffer, unsigned int key_val_buffer_size, realloc_callback_t* key_val_realloc)
{
    if (!is_table_exp10_filled)
        init_table_exp10();

    // consume the utf-8 BOM if it exists
    if (strncmp(input, "\xEF\xBB\xBF", 3) == 0)
    {
        input += 3;
        input_size -= 3;
    }

    json_context_t context;
    context.source = input;
    context.nest_depth = 0;
    context.string_buffer = string_buffer;
    context.string_buffer_size = string_buffer_size;
    context.string_buffer_brk = 0;
    context.string_buffer_realloc = string_realloc;
    context.key_val_buffer = key_val_buffer;
    context.key_val_buffer_size = key_val_buffer_size;
    context.key_val_buffer_brk = 0;
    context.key_val_buffer_realloc = key_val_realloc;

    json_value_t root;
    root.type = JSON_NULL;
    if (setjmp(context.error_jmp_buf) == 0)
    {
        parse_element(&root, &input, &context);
#ifndef JSON_IGNORE_TRAILING_GARBAGE
        parse_whitespace(&input, &context);
        if (input < context.source + input_size) // extra garbage
        {
            context.result.accepted = false;
            context.result.error.reason = "trailing garbage";
        }
        else
#endif
        {
            context.result.accepted = true;
            context.result.value = root;
        }

        return context.result;
    }
    else // error path
    {
        context.result.accepted = false;
        return context.result;
    }
}
