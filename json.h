#ifndef JSON_H
#define JSON_H

#include <stdbool.h>
#include <stdint.h>

typedef struct json_object_t
{
    unsigned int entry_count;
    unsigned int start_idx;  // in the keyval buffer
} json_object_t;

typedef struct json_array_t
{
    unsigned int entry_count;
    unsigned int start_idx;  // in the keyval buffer
} json_array_t;

typedef union json_number_t
{
    double num_real;
    uint64_t num_int;
} json_number_t;

typedef struct json_value_t
{
    enum
    {
        JSON_OBJECT,
        JSON_ARRAY,
        JSON_STRING,
        JSON_NUMBER,
        JSON_BOOL,
        JSON_NULL
    } type;
    bool is_real;
    union
    {
        json_object_t object;
        json_array_t array;
        struct
        {
            const char* string;
            unsigned str_len;
        };
        json_number_t num;
        bool boolean;
    };
} json_value_t;

typedef struct json_result_t
{
    bool accepted;
    union
    {
        json_value_t value;
        struct
        {
            const char* reason;
            int index; // TODO : replace with (line, col) pair
        } error;
    };
} json_result_t;

typedef struct json_key_value_t
{
    const char* key;
    struct json_value_t value;
} json_key_value_t;

typedef void*(realloc_callback_t)(void*, int);

json_result_t parse_json(const char* input, int input_size,
                         char* string_buffer, unsigned int string_buffer_size, realloc_callback_t* string_realloc,
                         json_key_value_t* key_val_buffer, unsigned int key_val_buffer_size, realloc_callback_t* key_val_realloc);

#endif // JSON_H
