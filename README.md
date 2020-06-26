# smoljson
Blazing fast and light SIMD JSON parser in a few hundreds lines of C Code

Parsing up to 4 GB of JSON data per second!

## What's smoljson?
smoljson is a single small .c file, single header C library aiming to parse JSON as fast as possible using x86 SIMD instruction sets up to AVX2 if available.

It's fully compliant to the JSON spec, tested using https://github.com/nst/JSONTestSuite, and has mostly on-par (sometimes better!) performance with the largest SIMD json parser library, simdjson (https://github.com/simdjson/simdjson).

It can also print JSON values to a string buffer, or create and manipualte JSON objects like every JSON library worthy of this name.

smoljson can work in zero-allocation mode : all you have to do, is supply buffers to the parsing function. Said buffers can be dynamically reallocated if you supply the matching callback.

smoljson also supports the JSON Pointer syntax to access data as described in https://tools.ietf.org/html/rfc6901 .

smoljson can also be thread safe if you uncomment the following define in json.c : "//#define JSON_STACK_WORKING_MEMORY"

## How to use it

The interface : 
```c
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
        JSON_NUMBER_INT,
        JSON_NUMBER_FLOAT,
        JSON_BOOL,
        JSON_NULL
    } type;
    union
    {
        json_object_t object;
        json_array_t array;
        struct
        {
            unsigned int str_idx;
            unsigned int str_len;
        };
        json_number_t num;
        bool boolean;
    };
} json_value_t;

typedef struct json_context_t
{
    char* string_buffer; unsigned int string_buffer_size; realloc_callback_t* string_realloc;
    json_key_value_t* key_val_buffer; unsigned int key_val_buffer_size; realloc_callback_t* key_val_realloc;
    unsigned int string_buffer_brk; unsigned int key_val_buffer_brk;
} json_context_t;
```
```c
json_result_t parse_json        (const char* input, int input_size, json_context_t* context);
unsigned int  print_json        (json_value_t* value, json_context_t* context, bool pretty, char* buffer, unsigned int buffer_size);
json_value_t *json_pointer      (const char* pointer, json_value_t* root, json_context_t* context);
const char   *json_get_string   (json_value_t* val, json_context_t* context);
json_value_t *json_create_string(const char* str, json_context_t* context);
json_value_t *json_create_value (json_context_t* context, uint8_t type);
void          json_object_add   (json_context_t* context, json_value_t* object, const char* key, json_value_t* value);
void          json_array_add    (json_context_t* context, json_value_t* object, json_value_t* value);
```

Parsing a json file:
```c
  #define KEYVAL_SIZE (65536*256)
  #define STRING_SIZE (65536*256)

  json_key_value_t keyval_buf[KEYVAL_SIZE];
  char string_buf[STRING_SIZE];

  const uint8_t* read_buffer = read_file("your_file.json");
  
  json_context_t context;
  context.string_buffer = string_buf; context.string_buffer_size = STRING_SIZE; context.string_realloc = NULL; // no dynamic reallocation
  context.key_val_buffer = keyval_buf, context.key_val_buffer_size = KEYVAL_SIZE; context.key_val_realloc = NULL;

  json_result_t result = parse_json(data, len, &context);
  if (!result.accepted)
  {
      printf("error : %s\n", result.error.reason);
  }
```
Printing it: 
```c
    static char print_buf[65536*128];
    print_json(&result.value, &context, false, print_buf, 65536*128);
    printf("out: %s\n", print_buf);
```

Manipulating json data:
```c
  json_value_t* value = json_pointer("/example_key", &result.value, &context);
  printf("value : %s\n", json_get_string(value, &context);
  
  json_value_t* root = json_create_value(context, JSON_OBJECT);
  json_object_add(context, root, "hello world key", json_create_string("hello world!", context));
  static char print_buf[65536*128];
  print_json(&result.value, &context, false, print_buf, 65536*128);
  printf("out: %s\n", print_buf);
  /* prints :
  * {
  * "hello world key" : "hello world!"
  * }
  */
```

## Performance

Benchmarks performed on Windows 10 with GCC 5.1.0 (-O3, -march=native) on an Intel 4690K @ 3.9GHz.

Values are parsing bandwith in MB/s, tests were performed with the test .json files in the test_files/ subfolder.

![bench1](https://i.imgur.com/mMadFKm.png)
![bench2](https://i.imgur.com/cOsuTck.png)
![bench3](https://i.imgur.com/6r11wlg.png)
