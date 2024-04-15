// to compile the code: gcc -O0 -g3 --std=c99 -o emb_host emb_host.c -g `dpu-pkg-config --cflags
// --libs dpu` to build a shared library: gcc -shared -Wl,-soname,emb_host -o emblib.so -fPIC
// emb_host.c `dpu-pkg-config --cflags --libs dpu`
#include "common.h"
#include "host/include/host.h" 
#include "emb_types.h"
#include "../aes.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#define RT_CONFIG 0

#ifndef DPU_BINARY
#    define DPU_BINARY "../upmem/emb_dpu_lookup" // Relative path regarding the PyTorch code
#endif

int32_t* buffer_data[NR_COLS];
int32_t* test;
struct dpu_set_t dpu_ranks[AVAILABLE_RANKS], dpu_r[AVAILABLE_RANKS];

#define TIME_NOW(_t) (clock_gettime(CLOCK_MONOTONIC, (_t)))

/**
 * @struct dpu_runtime
 * @brief DPU execution times
 */
typedef struct dpu_runtime_totals {
    double execution_time_prepare;
    double execution_time_populate_copy_in;
    double execution_time_copy_in;
    double execution_time_copy_out;
    double execution_time_aggregate_result;
    double execution_time_launch;
} dpu_runtime_totals;

/**
 * @struct dpu_timespec
 * @brief ....
 */
typedef struct dpu_timespec {
    long tv_nsec;
    long tv_sec;
} dpu_timespec;

/**
 * @struct dpu_runtime_interval
 * @brief DPU execution interval
 */
typedef struct dpu_runtime_interval {
    dpu_timespec start;
    dpu_timespec stop;
} dpu_runtime_interval;

/**
 * @struct dpu_runtime_config
 * @brief ...
 */
typedef enum dpu_runtime_config {
    RT_ALL = 0,
    RT_LAUNCH = 1
} dpu_runtime_config;

/**
 * @struct dpu_runtime_group
 * @brief ...
 */
typedef struct dpu_runtime_group {
    unsigned int in_use;
    unsigned int length;
    dpu_runtime_interval *intervals;
} dpu_runtime_group;

static void enomem() {
    fprintf(stderr, "Out of memory\n");
    exit(ENOMEM);
}

static void copy_interval(dpu_runtime_interval *interval,
                          struct timespec * const start,
                          struct timespec * const end) {
    interval->start.tv_nsec = start->tv_nsec;
    interval->start.tv_sec = start->tv_sec;
    interval->stop.tv_nsec = end->tv_nsec;
    interval->stop.tv_sec = end->tv_sec;
}

static int alloc_buffers(uint32_t table_id, int32_t *table_data, uint64_t nr_rows) {
    
    for(int j=0; j<NR_COLS; j++){

        size_t sz = nr_rows*sizeof(int32_t);
        buffer_data[j] = malloc(ALIGN(sz,8));
        //test[j] = malloc(ALIGN(sz,8));
        if (buffer_data[j] == NULL) {
            return ENOMEM;
        }

        for(int k=0; k<nr_rows; k++){
            buffer_data[j][k] = table_data[k*NR_COLS+j];
        }

    }

    return 0;
}

/*
    Params:
    0. table_id: embedding table number.
    1. nr_rows: number of rows of the embedding table
    2. NR_COLS: number of columns of the embedding table
    3. table_data: a pointer of the size nr_rows*NR_COLS containing table's data
    Result:
    This function breaks down each embedding table into chunks of maximum MAX_CAPACITY
    and pushes each chunk(buffer) to one dpu as well as number of rows and columns of the
    corresponding table with the index of the first and last row held in each dpu.
*/

void populate_mram(uint32_t table_id, uint64_t nr_rows, int32_t *table_data, dpu_runtime_totals *runtime){
    struct timespec start, end;
    clock_t start1, end1;
    double cpu_time_used;
    if(table_id>=AVAILABLE_RANKS){
        fprintf(stderr,"%d ranks available but tried to load table %dth",AVAILABLE_RANKS,table_id);
        exit(1);
    }

    //TIME_NOW(&start);
    if (alloc_buffers(table_id, table_data, nr_rows) != 0) {
        enomem();
    }
    //TIME_NOW(&end);

    //if (runtime) runtime->execution_time_prepare += TIME_DIFFERENCE(start, end);

    //TIME_NOW(&start);
    
    struct dpu_set_t set, dpu, dpu_rank;
    DPU_ASSERT(dpu_alloc(NR_COLS, NULL, &set));
    DPU_ASSERT(dpu_load(set, DPU_BINARY, NULL));
    //printf("set:%d\n", NR_COLS);
    //print(dpu)
    uint32_t len;
    uint8_t dpu_id,rank_id;
    //  start1 = clock();
    DPU_FOREACH(set, dpu, dpu_id){
        
        DPU_ASSERT(dpu_prepare_xfer(dpu, buffer_data[dpu_id]));
        //printf("set:%d dpu:%d \n", set, dpu_id);
    }
    DPU_ASSERT(dpu_push_xfer(set,DPU_XFER_TO_DPU, "emb_data", 0, ALIGN(nr_rows*sizeof(int32_t),8), DPU_XFER_DEFAULT));

    // end1 = clock();
    for (int i = 0; i < NR_COLS; i++){
        free(buffer_data[i]);
    }

    dpu_ranks[table_id] = set;
    // printf("set: %d:\n", set);
    //TIME_NOW(&end);

    //if (runtime) runtime->execution_time_populate_copy_in += TIME_DIFFERENCE(start, end);
    // Calculate the CPU time used
    // cpu_time_used = ((double) (end1 - start1)) / CLOCKS_PER_SEC;

    // Print the execution time
    // printf("CPU-DPU time for data: %f seconds\n", cpu_time_used);
    return;
}


/*
    Params:
    1. ans: a pointer that be updated with the rows that we lookup
    2. input: a pointer containing the specific rows we want to lookup
    3. length: contains the number of rows that we want to lookup from the table
    4. nr_rows: number of rows of the embedding table
    5. NR_COLS: number of columns of the embedding table
    Result:
    This function updates ans with the elements of the rows that we have lookedup
*/
int32_t* lookup(uint32_t* indices, uint32_t *offsets, uint64_t indices_len,
                uint64_t nr_batches, float *final_results, uint32_t table_id
                //,dpu_runtime_group *runtime_group
                ){
    //struct timespec start, end;
    int dpu_id;
    uint64_t copied_indices;
    struct dpu_set_t dpu;
    struct query_len lengths;
    clock_t start1, end1, start2 ,start3, start4, end4, end2, end3;
    double cpu_dpu_time, lunch_time, dpu_cpu_time, merge_time = 0 ;
    //if (runtime_group && RT_CONFIG == RT_ALL) TIME_NOW(&start);

    // uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    // struct AES_ctx ctx;
    // double st = clock();
    // printf("len in: %d\n", indices_len);
    // test = malloc(indices_len * sizeof(uint32_t));
    // printf("num bat: %d\n", nr_batches);
    // test_in = malloc(nr_batches * sizeof(uint32_t));
    // test_of = malloc(nr_batches * sizeof(uint32_t));
    // #pragma omp parallel for
    //  for(int j=0; j<nr_batches; j++){
    //     test[j] = 5;
    //  }
    //     AES_init_ctx(&ctx, key);
    //     AES_ECB_encrypt(&ctx, test); 
    //  double en = clock();
    //  double cpu_time = ((double) (en - st)) / CLOCKS_PER_SEC;
    //  printf("rand gen time: %f seconds\n", cpu_time);
     
    //  start1 = clock();
    DPU_ASSERT(dpu_prepare_xfer(dpu_ranks[table_id],indices));
    DPU_ASSERT(dpu_push_xfer(dpu_ranks[table_id],DPU_XFER_TO_DPU,"input_indices",0,ALIGN(
        indices_len*sizeof(uint32_t),8),DPU_XFER_DEFAULT));
    
    DPU_ASSERT(dpu_prepare_xfer(dpu_ranks[table_id],offsets));
    DPU_ASSERT(dpu_push_xfer(dpu_ranks[table_id],DPU_XFER_TO_DPU,"input_offsets",0,ALIGN(
        nr_batches*sizeof(uint32_t),8),DPU_XFER_DEFAULT));

    /*DPU_ASSERT(dpu_prepare_xfer(dpu_ranks[table_id],indices));
    DPU_ASSERT(dpu_push_xfer(dpu_ranks[table_id],DPU_XFER_TO_DPU,"input_indices",0,ALIGN(
        indices_len*sizeof(uint32_t),8),DPU_XFER_DEFAULT));*/

    lengths.indices_len=indices_len;
    lengths.nr_batches=nr_batches;
    // printf("ind len: %d\n", indices_len);
    // printf("batch len: %d\n", nr_batches);
    DPU_ASSERT(dpu_prepare_xfer(dpu_ranks[table_id],&lengths));
    DPU_ASSERT(dpu_push_xfer(dpu_ranks[table_id],DPU_XFER_TO_DPU,"input_lengths",0,
    sizeof(struct query_len),DPU_XFER_DEFAULT));
    // end1 = clock();
    //cpu_dpu_time += ((double) (end1 - start1)) / CLOCKS_PER_SEC;
    // start2 = clock();
    DPU_ASSERT(dpu_launch(dpu_ranks[table_id], DPU_SYNCHRONOUS));
    //can we run this async to do post-processing?
    // end2 = clock(); 
     lunch_time += ((double) (end2 - start2)) / CLOCKS_PER_SEC; 
    /* if (runtime_group && RT_CONFIG == RT_LAUNCH) {
        if(runtime_group[table_id].in_use >= runtime_group[table_id].length) {
            TIME_NOW(&end);
            fprintf(stderr,
                "ERROR: (runtime_group[%d].in_use) = %d >= runtime_group[%d].length = %d\n",
                dpu_id, runtime_group[table_id].in_use, table_id, runtime_group[table_id].length);
            exit(1);
        }
        copy_interval(
            &runtime_group->intervals[runtime_group[table_id].in_use], &start, &end);
            runtime_group[table_id].in_use++;
    } */
    // start3 = clock();
    int32_t tmp_results[NR_COLS][nr_batches];
    DPU_FOREACH(dpu_ranks[table_id], dpu, dpu_id){
        DPU_ASSERT(dpu_prepare_xfer(dpu,&tmp_results[dpu_id][0]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_ranks[table_id], DPU_XFER_FROM_DPU, "results",0,
    ALIGN(sizeof(int32_t)*nr_batches,8), DPU_XFER_DEFAULT));
    // end3 = clock();
    // dpu_cpu_time += ((double) (end3 - start3)) / CLOCKS_PER_SEC;
    // start4 = clock();

    for (int j=0; j<NR_COLS; j++){
        for(int i=0; i<nr_batches; i++)
            final_results[i*NR_COLS+j]=(float)tmp_results[j][i]/pow(10,9);
    }
    // end4 = clock();
    // merge_time = ((double) (end4 - start4)) / CLOCKS_PER_SEC;
    // printf("CPU-DPU time for others: %f seconds\n", cpu_dpu_time);
    // printf("lunch time: %f seconds\n", lunch_time);
    // printf("DPU-CPU time: %f seconds\n", dpu_cpu_time);
    // printf("Merge time: %f seconds\n", merge_time);

    
    return 0;
}

// int32_t* lookup(uint32_t* indices, uint32_t *offsets, uint64_t indices_len,
//                 uint64_t nr_batches
//                 //,dpu_runtime_group *runtime_group
//                 ){
//     //struct timespec start, end;
//     int dpu_id;
//     uint64_t copied_indices;
//     struct dpu_set_t dpu;
//     struct query_len lengths;
//     clock_t start1, end1, start2 ,start3, start4, end4, end2, end3;
//     double cpu_dpu_time, lunch_time, dpu_cpu_time, merge_time = 0 ;
//     //if (runtime_group && RT_CONFIG == RT_ALL) TIME_NOW(&start);

//     uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
//     struct AES_ctx ctx;
//     double st = clock();
//     // printf("len in: %d\n", indices_len);
//     // test = malloc(indices_len * sizeof(uint32_t));
//     // printf("num bat: %d\n", nr_batches);
//     // test_in = malloc(nr_batches * sizeof(uint32_t));
//     // test_of = malloc(nr_batches * sizeof(uint32_t));
//     // #pragma omp parallel for
//     //  for(int j=0; j<nr_batches; j++){
//     //     test[j] = 5;
//     //  }
//     //     AES_init_ctx(&ctx, key);
//     //     AES_ECB_encrypt(&ctx, test); 
//     //  double en = clock();
//     //  double cpu_time = ((double) (en - st)) / CLOCKS_PER_SEC;
//     //  printf("rand gen time: %f seconds\n", cpu_time);
     
//       start1 = clock();
//     DPU_FOREACH(dpu_ranks, table_id){
//         DPU_ASSERT(dpu_prepare_xfer(table_id,indices));
//     }
//     DPU_ASSERT(dpu_push_xfer(dpu_ranks,DPU_XFER_TO_DPU,"input_indices",0,ALIGN(
//         indices_len*sizeof(uint32_t),8),DPU_XFER_DEFAULT));
//     DPU_FOREACH(dpu_ranks,table_id){
//         DPU_ASSERT(dpu_prepare_xfer(table_id,offsets));
//     }
//     DPU_ASSERT(dpu_push_xfer(dpu_ranks,DPU_XFER_TO_DPU,"input_offsets",0,ALIGN(
//         nr_batches*sizeof(uint32_t),8),DPU_XFER_DEFAULT));

//     /*DPU_ASSERT(dpu_prepare_xfer(dpu_ranks[table_id],indices));
//     DPU_ASSERT(dpu_push_xfer(dpu_ranks[table_id],DPU_XFER_TO_DPU,"input_indices",0,ALIGN(
//         indices_len*sizeof(uint32_t),8),DPU_XFER_DEFAULT));*/

//     lengths.indices_len=indices_len;
//     lengths.nr_batches=nr_batches;
//      printf("ind len: %d\n", indices_len);
//     printf("batch len: %d\n", nr_batches);
//     DPU_FOREACH(dpu_ranks,table_id){
//         DPU_ASSERT(dpu_prepare_xfer(table_id,&lengths));
//     }
//     DPU_ASSERT(dpu_push_xfer(dpu_ranks,DPU_XFER_TO_DPU,"input_lengths",0,
//     sizeof(struct query_len),DPU_XFER_DEFAULT));

//     end1 = clock();
//     cpu_dpu_time += ((double) (end1 - start1)) / CLOCKS_PER_SEC;
//     printf("CPU-DPU time for others: %f seconds\n", cpu_dpu_time);

//     return 0;
// }
int
main() {
}