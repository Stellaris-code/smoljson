#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "json.h"

clock_t start, end;
double cpu_time_used;

#define KEYVAL_SIZE (65536*256)
#define STRING_SIZE (65536*256)

json_key_value_t keyval_buf[KEYVAL_SIZE];
char string_buf[STRING_SIZE];

#define BENCHMARK

int testFile(const char *filename) {

    FILE *f=fopen(filename,"rb");
    if(f == NULL) { return false; };
    fseek(f,0,SEEK_END);
    long len=ftell(f);
    fseek(f,0,SEEK_SET);
    char *data=(char*)malloc(len+1+4096); // extra space to be safe regarding SIMD operations
    fread(data,1,len,f);
    data[len]='\0';
    fclose(f);

    json_context_t context;
    context.string_buffer = string_buf; context.string_buffer_size = STRING_SIZE; context.string_realloc = NULL;
    context.key_val_buffer = keyval_buf, context.key_val_buffer_size = KEYVAL_SIZE; context.key_val_realloc = NULL;

#if defined(BENCHMARK)

    json_result_t result;
    start = clock();
    for (int i = 0; i < 10000; ++i)
    {
        result = parse_json(data, len, &context);
        if (!result.accepted)
        {
            printf("error : %s\n", result.error.reason);
        }
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("elapsed time : %f \n", cpu_time_used);
    printf("bandwidth : %f MB/s\n", len/1000000.0*10000/cpu_time_used);

#if 1

    static char print_buf[65536*128];
    unsigned int print_len;
    start = clock();
    for (int i = 0; i < 1000; ++i)
    {
        print_len = print_json(&result.value, &context, false, print_buf, 65536*128);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("stringify elapsed time : %f \n", cpu_time_used);
    printf("stringify bandwidth : %f MB/s\n", print_len/1000000.0*1000/cpu_time_used);
#endif

    free(data);

    return 1;

#else
    json_result_t result = parse_json(data, len, &context);
    if (!result.accepted)
    {
        printf("error : %s\n", result.error.reason);
    }

#if 1
    static char print_buf[65536*128];
    print_json(&result.value, &context, false, print_buf, 65536*128);
    printf("out: %s\n", print_buf);

    json_value_t* value = json_pointer("/", &result.value, &context);
    print_json(value, &context, true, print_buf, 65536);
    printf("value found : %s\n", print_buf);
#endif

    free(data);
    return result.accepted;
#endif
}

int main(int argc, const char * argv[]) {

    const char* path = argv[1];
    //path = "canada.json";
    //path = "citm_catalog.json";
    //path = "twitter.json";
    //path = "google_maps_api_response.json";
    //path = "test.json";

    int result = testFile(path);

    printf("-- result: %d\n", result);

    if (result == true) {
        return 0;
    } else {
        return 1;
    }
}
