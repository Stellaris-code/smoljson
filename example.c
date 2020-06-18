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

int testFile(const char *filename) {

    FILE *f=fopen(filename,"rb");
    if(f == NULL) { return false; };
    fseek(f,0,SEEK_END);
    long len=ftell(f);
    fseek(f,0,SEEK_SET);
    char *data=(char*)malloc(len+1);
    fread(data,1,len,f);
    data[len]='\0';
    fclose(f);

#if 0

    start = clock();
    for (int i = 0; i < 1000; ++i)
    {
        json_result_t result = parse_json(data, len,
                                          string_buf, STRING_SIZE, NULL,
                                          keyval_buf, KEYVAL_SIZE, NULL);
        if (!result.accepted)
        {
            printf("error : %s\n", result.error.reason);
        }
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("elapsed time : %f \n", cpu_time_used);
    printf("bandwidth : %f MB/s\n", len/1000000.0*1000/cpu_time_used);
    free(data);

    return 1;

#else
    json_result_t result = parse_json(data, len,
                                      string_buf, STRING_SIZE, NULL,
                                      keyval_buf, KEYVAL_SIZE, NULL);
    if (!result.accepted)
    {
        printf("error : %s\n", result.error.reason);
    }
    free(data);
    return result.accepted;
#endif
}

int main(int argc, const char * argv[]) {

    const char* path = argv[1];
    //path = "canada.json";
    //path = "citm_catalog.json";
    //path = "twitter.json";
    //path = "test.json";

    int result = testFile(path);

    printf("-- result: %d\n", result);

    if (result == true) {
        return 0;
    } else {
        return 1;
    }
}
