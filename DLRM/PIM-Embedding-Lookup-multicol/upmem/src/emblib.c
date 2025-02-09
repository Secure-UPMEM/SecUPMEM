#include "embedding.h"
#include "aes.h"
#include "fifo.h"

#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include <ctime>

/**
 *  @brief alloc synthetic embedding tables on HCPU side
 *  @param embedding_info information about embedding configuration
 *  @return allocated embedding tables
 */
int32_t **
alloc_emb_tables(embedding_info *emb_info) {
    int32_t **emb_tables = (int32_t **) malloc(emb_info->nr_embedding * sizeof(int32_t *));
    for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
         embedding_index++) {
        /* allocate embedding table on host side */
        emb_tables[embedding_index] =
            (int32_t *) malloc(emb_info->nr_rows * emb_info->nr_cols * sizeof(int32_t));
    }
    return emb_tables;
}

/**
 *  @brief free embedding tables on HCPU side
 *  @param emb_tables embedding tables
 *  @param embedding_info information about embedding configuration
 */
void
free_emb_tables(int32_t **emb_tables, embedding_info *emb_info) {
    for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
         embedding_index++) {
        /* free embedding table */
        free(emb_tables[embedding_index]);
    }
    free(emb_tables);
}
/** @brief type of data generation */
enum emb_datagen { RAND, CPT, ZERO, NONE };

/**
 *  @brief populate embedding table with synthetic data of 3 different kind (random, reference
 * counter, zeros )
 *  @param rank_mapping information about rank embedding mapping
 *  @param emb_info information about embedding configuration
 *  @param emb_tables embedding tables
 */
static int32_t cpt = 0;
static int32_t sign = 1;
void
synthetic_populate(embedding_rank_mapping *rank_mapping, embedding_info *emb_info,
                   input_info *i_info, int32_t **emb_tables) {

    uint64_t emb_data_type = CPT;

    printf("generate synthetic tables\n");
    if (emb_data_type == RAND) {
        for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
             embedding_index++) {
            /* synthetize random embedding table parameters */
            for (int i = 0; i < emb_info->nr_rows * emb_info->nr_cols; i++) {
                double data_norm = (double) (rand()) / ((double) RAND_MAX + 1) / i_info->nr_indexes;
                emb_tables[embedding_index][i] = (int32_t) (INT32_MAX * data_norm);
            }
        }
    } else if (emb_data_type == CPT) {
        for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
             embedding_index++) {
            for (int i = 0; i < emb_info->nr_rows * emb_info->nr_cols; i++) {
                emb_tables[embedding_index][i] = cpt * sign;
                cpt++;
                if ((float) (cpt) > (float) INT32_MAX / i_info->nr_indexes)
                    cpt = 0;
                // printf("cpt %d\n", emb_tables[embedding_index][i]);

                sign = sign * -1;
            }
        }
    } else if (emb_data_type == ZERO) {
        for (uint64_t embedding_index = 0; embedding_index < emb_info->nr_embedding;
             embedding_index++) {
            for (int i = 0; i < emb_info->nr_rows * emb_info->nr_cols; i++) {
                emb_tables[embedding_index][i] = (int32_t) (0);
            }
        }
    }

    printf("populate mram with embedding synthetic tables\n");

    /* store one embedding to DPU MRAM */
    populate_mram(rank_mapping, emb_info, emb_tables);
}

/**
 *   @brief check_one_embedding_set_inference
 *   @param embedding_index_start  start index in embedding array
 *   @param emb_tables embedding tables
 *   @param indices embedding input indices
 *   @param offsets embedding input offset
 *   @param indices_len embedding input lendgth
 *   @param nr_batches number of batches for input
 *   @param nr_cols embedding number of cols
 *   @param cpu_results CPU result buffer
 *   @param nr_embedding number of embedding
 */
void
cpu_embedding_lookup(uint64_t embedding_index_start, int32_t **emb_tables, uint32_t **indices,
                     uint32_t **offsets, uint64_t *indices_len, uint64_t nr_batches,
                     uint64_t nr_cols, float ***cpu_results, uint64_t nr_embedding) {

    uint64_t index = 0;

    /* for each embedding */
    for (uint64_t embedding_index = embedding_index_start;
         embedding_index < nr_embedding + embedding_index_start; embedding_index++) {
        /* for each input batch of index */
        for (int batch_index = 0; batch_index < nr_batches; batch_index++) {
            /* reset tmb buffer */
            for (int col_index = 0; col_index < nr_cols; col_index++)
                cpu_results[embedding_index][batch_index][col_index] = 0;

            /* check limits */
            uint64_t upper_bound = batch_index == nr_batches - 1 ?
                                       indices_len[embedding_index] :
                                       offsets[embedding_index][batch_index + 1];
            for (uint64_t ind_ptr = offsets[embedding_index][batch_index]; ind_ptr < upper_bound;
                 ind_ptr++) {
                /* solve ind_ptr */
                index = indices[embedding_index][ind_ptr];
                for (int col_index = 0; col_index < nr_cols; col_index++) {
                    /*Embedding reduction mode : ADD */
                    cpu_results[embedding_index][batch_index][col_index] +=
                        emb_tables[embedding_index][index * nr_cols + col_index];
                }
            }
        }
    }
}

typedef struct {
    uint64_t embdedding_index;
    int32_t **emb_tables;
    uint32_t **indices;
    uint32_t **offsets;
    uint64_t *indices_len;
    uint64_t nr_batches;
    uint64_t nr_cols;
    uint64_t nr_embedding;
    float ***cpu_results;
} param_map_t;

/**
 *   @brief thread function that call kernel function to check dpu results
 *   @param argv thread args
 */
void *
thread_cpu_embedding_lookup(void *argv) {
    param_map_t *param = (param_map_t *) argv;
    cpu_embedding_lookup(param->embdedding_index, param->emb_tables, param->indices, param->offsets,
                         param->indices_len, param->nr_batches, param->nr_cols, param->cpu_results,
                         param->nr_embedding);

    pthread_exit(NULL);
}
pthread_t THREAD_MAP[MAX_NR_EMBEDDING];
#define CPU_NR_THREAD_MAX 100

/** @brief check DPU embedding inference result for each embedding and each batch
 *  @param nr_thread number of thread for CPU lookup
 *  @param emb_tables host side embeding tables
 *  @param nr_embedding number of embedding in emb_tables
 *  @param indices array that stores indices [EMB_INDEX][BATCH_INDEX][INDEXES]
 *  @param offsets array that stores indices offset (pytorch EmbedingBag convention)
 *  [EMB_INDEX][BATCH_INDEX][OFFSET]
 *  @param indices_len  gives the lenght of the input indices vector for each embedding [EMB_INDEX]
 *  @param nr_batches gives the number of batch (same for each embedding) in indices
 *  @param nr_cols Embedding Number of columns (same for each embedding)
 *  @param cpu_results CPU results buffer
 */
void
cpu_lookup(uint64_t nr_thread, int32_t **emb_tables, uint64_t nr_embedding, uint32_t **indices,
           uint32_t **offsets, uint64_t *indices_len, uint64_t nr_batches, uint64_t nr_cols,
           float ***cpu_results) {

#define CPU_MT 1
#if (CPU_MT == 1)
    param_map_t **param = malloc(sizeof(param_map_t *) * nr_embedding);
    uint64_t emb_lot = nr_embedding / nr_thread;
    for (uint64_t embdedding_index = 0; embdedding_index < nr_embedding; embdedding_index++)
        param[embdedding_index] = (param_map_t *) malloc(sizeof(param_map_t));

    for (uint64_t embdedding_index = 0, emb_lot_index = 0; emb_lot_index < nr_thread;
         embdedding_index += emb_lot, emb_lot_index++) {

        param[emb_lot_index]->embdedding_index = embdedding_index;
        param[emb_lot_index]->emb_tables = emb_tables;
        param[emb_lot_index]->indices = indices;
        param[emb_lot_index]->offsets = offsets;
        param[emb_lot_index]->indices_len = indices_len;
        param[emb_lot_index]->nr_batches = nr_batches;
        param[emb_lot_index]->nr_cols = nr_cols;
        param[emb_lot_index]->nr_embedding = emb_lot;
        param[emb_lot_index]->cpu_results = cpu_results;

        pthread_create(&THREAD_MAP[emb_lot_index], NULL, thread_cpu_embedding_lookup,
                       param[emb_lot_index]);
    }

    for (int i = 0; i < nr_thread; i++)
        pthread_join(THREAD_MAP[i], NULL);

    for (int i = 0; i < nr_embedding; i++)
        free(param[i]);

    free(param);
#else
    param_map_t **param = malloc(sizeof(param_map_t *) * 1);

    param[emb_lot_index]->embdedding_index = 0;
    param[emb_lot_index]->emb_tables = emb_tables;
    param[emb_lot_index]->indices = indices;
    param[emb_lot_index]->offsets = offsets;
    param[emb_lot_index]->indices_len = indices_len;
    param[emb_lot_index]->nr_batches = nr_batches;
    param[emb_lot_index]->nr_cols = nr_cols;
    param[emb_lot_index]->nr_embedding = nr_embedding;
    param[emb_lot_index]->cpu_results = cpu_results;

    pthread_create(&THREAD_MAP[0], NULL, thread_cpu_embedding_lookup, param[0]);
    pthread_join(THREAD_MAP[0], NULL);

    for (int i = 0; i < 1; i++)
        free(param[i]);
    free(param);

#endif
}

/**
 *  @brief allocate input indices buffer
 *  @param nr_embedding number of embedding
 *  @param nr_batches batch size (number of indice vector per batch)
 *  @param indices_per_batch number of indices per indice vector
 *  @return buffer
 */
uint32_t **
alloc_indices_buffer(uint64_t nr_embedding, uint64_t nr_batches, uint64_t indices_per_batch) {
    uint32_t **indices = (uint32_t **) malloc(nr_embedding * sizeof(uint32_t *));
    for (uint32_t k = 0; k < nr_embedding; k++) {
        indices[k] = (uint32_t *) malloc(nr_batches * indices_per_batch * sizeof(uint32_t));
    }
    return indices;
}

/**
 *  @brief free input indices buffer
 *  @param indices input indices buffer
 *  @param nr_embedding number of embedding
 */
void
free_indices_buffer(uint32_t **indices, uint64_t nr_embedding) {
    for (uint32_t k = 0; k < nr_embedding; k++) {
        free(indices[k]);
    }
    free(indices);
}

/**
 *  @brief allocate offset buffer
 *  @param nr_embedding number of embedding
 *  @param nr_batches batch size (number of indice vector per batch)
 *  @return buffer
 */
uint32_t **
alloc_offset_buffer(uint64_t nr_embedding, uint64_t nr_batches) {
    uint32_t **offsets = (uint32_t **) malloc(nr_embedding * sizeof(uint32_t *));
    for (uint32_t k = 0; k < nr_embedding; k++) {
        offsets[k] = (uint32_t *) malloc(nr_batches * sizeof(uint32_t));
    }
    return offsets;
}

/**
 *  @brief free input offset buffer
 *  @param offsets input offset buffer
 *  @param nr_embedding number of embedding
 */
void
free_offset_buffer(uint32_t **offsets, uint64_t nr_embedding) {
    for (uint32_t k = 0; k < nr_embedding; k++) {
        free(offsets[k]);
    }
    free(offsets);
}

/**
 *  @brief alloc input info structure
 *  @param nr_embedding number of embedding
 *  @param nr_batches batch size (number of indice vector per batch)
 *  @param indices_per_batch number of indices per indice vector
 *  @return input info structure
 */
input_info *
alloc_input_info(uint64_t nr_embedding, uint64_t nr_batches, uint64_t indices_per_lookup) {

    struct input_info *info = malloc(sizeof(struct input_info));
    info->nr_indexes = indices_per_lookup;
    info->indices_len = (uint64_t *) malloc(MAX_NR_EMBEDDING * sizeof(uint64_t));
    info->nr_batches = nr_batches;

    return info;
}

/**
 *  @brief free input info structure
 *  @param info input info structure
 */
void
free_input_info(struct input_info *info) {
    free(info->indices_len);
}

/**
 *  @brief allocate embedding info structure
 *  @param nr_embedding number of embedding
 *  @param nr_rows number of rows for each embedding
 *  @param nr_cols number of cols for each embedding
 *  @param sizeT embedding data size (byte)
 *  @return embedding info structure
 */
embedding_info *
alloc_embedding_info(uint32_t nr_embedding, uint32_t nr_rows, uint32_t nr_cols, uint32_t sizeT) {
    embedding_info *emb_info = malloc(sizeof(embedding_info));
    emb_info->nr_embedding = nr_embedding;
    emb_info->nr_rows = nr_rows;
    emb_info->nr_cols = nr_cols;
    emb_info->sizeT = sizeT;
    return emb_info;
};

/**
 *  @brief free embedding info structure
 *  @param emb_info embedding info structure
 */
void
free_embedding_info(embedding_info *emb_info) {
    free(emb_info);
}

/**
 *  @brief computes sizes of synthetic input structure
 *  @param input_info input info structure
 *  @param indices_per_batch input indices buffer
 *  @param nr_embedding number of embedding
 *  @param nr_batches batch size (number of indice vector per batch)
 *  @param nr_rows embedding number of rows
 */
void
build_synthetic_input_size(struct input_info *input_info, uint32_t **indices_per_batch,
                           uint64_t nr_embedding, uint64_t nr_batches, uint64_t nr_rows) {
    uint32_t indices_per_lookup;
    for (uint64_t embedding_index = 0; embedding_index < nr_embedding; embedding_index++) {

        input_info->nr_batches = nr_batches;
        input_info->indices_len[embedding_index] = 0;
        for (uint64_t batch_index = 0; batch_index < nr_batches; batch_index++) {
#if (RAND_INPUT_SIZE == 1)
            double indices_per_lookup_norm = ((double) rand() / RAND_MAX);
            indices_per_lookup = (uint32_t) (indices_per_lookup_norm * INDICES_PER_LOOKUP_RAND);
#else
            indices_per_lookup = INDICES_PER_LOOKUP;
#endif
            indices_per_batch[embedding_index][batch_index] = indices_per_lookup;
            input_info->indices_len[embedding_index] += indices_per_lookup;
        }
    }
}

/**
 *  @brief computes sizes of synthetic input structure
 *  @param indices input indices buffer
 *  @param offsets input offsers buffer
 *  @param input_info input info structure
 *  @param nr_embedding number of embedding
 *  @param nr_batches batch size (number of indice vector per batch)
 *  @param indices_per_batch input indices per batch
 *  @param nr_rows number of embedding rows
 *  @param nr_cols number of embedding cols
 */
void
build_synthetic_input_data(uint32_t **indices, uint32_t **offsets, struct input_info *input_info,
                           uint64_t nr_embedding, uint64_t nr_batches, uint32_t **indices_per_batch,
                           uint64_t nr_rows, uint64_t nr_cols) {

    for (uint64_t k = 0; k < nr_embedding; k++) {
        input_info->nr_batches = nr_batches;
        input_info->indices_len[k] = 0;
        for (uint64_t batch_index = 0; batch_index < nr_batches; batch_index++)
            input_info->indices_len[k] += indices_per_batch[k][batch_index];

        offsets[k][0] = 0;
        for (uint64_t batch_index = 1; batch_index < nr_batches; batch_index++)
            offsets[k][batch_index] =
                offsets[k][batch_index - 1] + indices_per_batch[k][batch_index - 1];

        for (uint64_t batch_index = 0; batch_index < input_info->nr_batches; batch_index++) {
            for (uint64_t j = 0; j < indices_per_batch[k][batch_index]; j++) {
                double index_norm = ((double) rand() / RAND_MAX);
                uint64_t index = (uint64_t) (nr_rows * index_norm);
                indices[k][offsets[k][batch_index] + j] = index;
                assert(index < nr_rows);
            }
        }
    }
}

/**
 *  @brief perform DPU embedding table inference given input indices with multiple embedding
 *  @param indices input indices buffer
 *  @param offsets input offsers buffer
 *  @param input_info input info structure
 *  @param rank_mapping_info rank embedding mapping structure
 *  @param emb_tables embedding table buffer
 *  @param cpu_results cpu_results buffer
 *  @param result_buffer DPU formated result buffer
 *  @param dpu_result_buffer DPU result buffer
 *  @param emb_info embedding info
 */
void
synthetic_inference(uint32_t **indices, uint32_t **offsets, input_info *input_info,
                    embedding_rank_mapping *rank_mapping_info, int32_t **emb_tables,
                    float ***cpu_results, float **result_buffer, int32_t **dpu_result_buffer,
                    embedding_info *emb_info) {

    __attribute__((unused)) bool valid = true;
    uint64_t nr_embedding = emb_info->nr_embedding;
    uint64_t nr_batches = input_info->nr_batches;
    uint64_t nr_rows = emb_info->nr_rows;
    uint64_t nr_cols = emb_info->nr_cols;
    double cpu_p_ratio = 0, dpu_p_ratio = 0;
    double cpu_time = 0, dpu_time = 0;
    uint64_t multi_run = 20;
    struct query_len *lengths = malloc(nr_embedding * sizeof(struct query_len));

    for (int i = 0; i < multi_run; i++) {

        struct timespec start_time, start_process_time, stop_time, stop_process_time;
        clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_process_time);

        /* inference */
        lookup(indices, lengths, offsets, input_info, rank_mapping_info, nr_embedding, nr_cols, nr_rows,
               result_buffer, dpu_result_buffer);
        clock_gettime(CLOCK_MONOTONIC_RAW, &stop_time);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop_process_time);
        double time = (float) ((stop_time.tv_sec - start_time.tv_sec) * 1e9 + stop_time.tv_nsec -
                               start_time.tv_nsec) /
                      (1e6);

        double process_time =
            (float) ((stop_process_time.tv_sec - start_process_time.tv_sec) * 1e9 +
                     stop_process_time.tv_nsec - start_process_time.tv_nsec) /
            (1e6);

        // printf("[PERF] DPU CLOCK RAW time [ms]: %5.2f\n"
        //        "[PERF] DPUTIME time [ms]: %5.2f\n"
        //        "[PERF] DPU PROCESS ratio: %5.2f\n",
        //        time, process_time, process_time / time * 100.0);

        dpu_p_ratio += process_time / time * 100.0;
        dpu_time += time;
    }

    free(lengths);

    double dpu_time_ms = dpu_time / multi_run;

    //The parallel cpu computation
    
    // printf("nr_embedding: %ld", nr_embedding);
    // // printf("indices_len: %d", input_info->indices_len);
    // printf("nr_batches: %ld", nr_batches);
    // printf("nr_cols: %ld", nr_cols);
    // printf("nr_rows: %ld", nr_rows);
    uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	struct AES_ctx ctx;
    AES_init_ctx(&ctx, key);
    
    uint8_t *** cpupart_result;
	cpupart_result = malloc(nr_embedding * sizeof(uint8_t**));
    for(int i = 0; i < nr_embedding; i++){
        cpupart_result[i] = malloc(nr_batches * sizeof(uint8_t*));
        for(int j = 0; j < nr_batches ; j++){
            cpupart_result[i][j] = malloc(nr_cols * sizeof(uint8_t));
        }
    }
    int idx_per_row = input_info->indices_len[0]/nr_batches;
    uint8_t* first = malloc(input_info->indices_len[0]* nr_cols * sizeof(uint8_t));

    clock_t start = clock();
    // for(int i = 0; i < nr_embedding; i++){
    //     for(int j = 0; j < nr_batches ; j++){
    //         for(int k = 0; k < idx_per_row * nr_cols; k++){
    //             first[k] = (uint8_t)(20+(j*sizeof(uint8_t)));
    //         }
	//         AES_ECB_encrypt(&ctx, first);
    //         for(int k = 0; k < idx_per_row; k++){
    //             for(int in = 0; in < nr_cols; in++){
    //                 cpupart_result[i][j][in] += first[k*nr_cols + in];
    //             }
    //         }
    //     }
    // }
    for(int i = 0; i < nr_embedding; i++){
        for(int j = 0; j < input_info->indices_len[0] ; j++){
            first[j] = (uint8_t)(20 + (j * sizeof(uint8_t)));;
        }
        AES_ECB_encrypt(&ctx, first);
        for(int j = 0; j < nr_batches ; j++){
            for(int k = 0; k < idx_per_row; k++){
                int offset = j * idx_per_row + k;
                for(int in = 0; in < nr_cols; in++){
                    cpupart_result[i][j][in] += first[offset * nr_cols + in];  
                }
            }
        }
    }
    clock_t end = clock();
    double elapsed_cpu = ((double)(end - start) / CLOCKS_PER_SEC) * 1000;
    printf("cpu part: %f \n", elapsed_cpu);
    free(first);


    uint8_t *** precomputed_result;
	precomputed_result = malloc(nr_embedding * sizeof(uint8_t**));
    for(int i = 0; i < nr_embedding; i++){
        precomputed_result[i] = malloc(nr_batches * sizeof(uint8_t*));
        for(int j = 0; j < nr_batches ; j++){
            precomputed_result[i][j] = malloc(nr_cols * sizeof(uint8_t));
        }
    }
    // int idx_per_row = input_info->indices_len[0]/nr_batches;
    uint8_t* first1 = malloc(nr_batches* nr_cols * sizeof(uint8_t));

    clock_t start1 = clock();
    for(int i = 0; i < nr_embedding; i++){
        for(int j = 0; j < nr_batches * nr_cols ; j++){
            // for(int in = 0; in < nr_cols; in++){
            first1[j] = (uint8_t)(30 + (j * sizeof(uint8_t)));
        }
        AES_ECB_encrypt(&ctx, first1);
        for(int j = 0; j < nr_batches ; j++){
                for(int in = 0; in < nr_cols; in++){
                    cpupart_result[i][j][in] += first1[j * nr_cols + in];  
                }
            
        }
    }
    clock_t end1 = clock();
    double elapsed_cpu1 = ((double)(end1 - start1) / CLOCKS_PER_SEC) * 1000;
    printf("cpu precomp part: %f \n", elapsed_cpu1);
    free(first1);


    clock_t start_merge = clock();
    for(int i = 0; i < nr_embedding; i++){
        for(int j = 0; j < nr_batches ; j++){
            for(int in = 0; in < nr_cols; in++){
                cpupart_result[i][j][in] += result_buffer[i][j * nr_cols + in];
            }
        }
    }
    clock_t end_merge = clock();
    double elapsed_merge = ((double)(end_merge - start_merge) / CLOCKS_PER_SEC) * 1000;
    printf("merge: %f \n", elapsed_merge);

    int32_t* veriftags = malloc(nr_embedding * sizeof(int32_t));
    // srand(1);
    for(int i=0; i<nr_embedding; i++){
        veriftags[i] = rand();
    }
    clock_t start_verif = clock();
    int s1 =10;
    for(int i=0; i<nr_embedding; i++){
        int temp_verif = 0;
        int power = 1;
        for(int in = 0; in < nr_cols; in++){
            for(int j = 0; j < nr_batches ; j++){
                temp_verif += cpupart_result[i][j][in] * power; 
                power = (power * s1);
            }
        // if(temp_verif == veriftags[i]) //some precomputed tags to compare with
        //         printf("verified");
        }
    }

    clock_t end_verif = clock();
    double elapsed_verif= ((double)(end_verif - start_verif) / CLOCKS_PER_SEC) * 1000;
    printf("verif: %f \n", elapsed_verif);

    //end cpu parrallel
    
    {
        uint64_t cpu_nr_thread = CPU_NR_THREAD_MAX;
        if (cpu_nr_thread > nr_embedding)
            cpu_nr_thread = nr_embedding;
        else {
            while (nr_embedding % cpu_nr_thread)
                cpu_nr_thread--;
        }

        for (int i = 0; i < multi_run; i++) {
            struct timespec start_time, start_process_time, stop_time, stop_process_time;
            clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_process_time);

            cpu_lookup(cpu_nr_thread, emb_tables, nr_embedding, indices, offsets,
                       input_info->indices_len, nr_batches, nr_cols, cpu_results);
            clock_gettime(CLOCK_MONOTONIC_RAW, &stop_time);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stop_process_time);
            double time = (float) ((stop_time.tv_sec - start_time.tv_sec) * 1e9 +
                                   stop_time.tv_nsec - start_time.tv_nsec) /
                          (1e6);
            double process_time =
                (float) ((stop_process_time.tv_sec - start_process_time.tv_sec) * 1e9 +
                         stop_process_time.tv_nsec - start_process_time.tv_nsec) /
                (1e6);
            // printf("[PERF] CPU CLOCK RAW time [ms]: %5.2f\n"
            //        "[PERF] CPUTIME time [ms]: %5.2f\n"
            //        "[PERF] CPU PROCESS ratio: %5.2f\n",
            //        time, process_time, process_time / time * 100.0);

            cpu_p_ratio += process_time / time * 100.0;
            cpu_time += time;
#define CHECK_RESULTS 1
#if (CHECK_RESULTS == 1)
            {

                /* for each embedding */
                for (uint64_t embedding_index = 0; embedding_index < nr_embedding;
                     embedding_index++) {
                    /* for each input batch of index */
                    for (int batch_index = 0; batch_index < nr_batches; batch_index++) {

                        for (int col_index = 0; col_index < nr_cols; col_index++) {

                            float dpu_result =
                                result_buffer[embedding_index][batch_index * nr_cols + col_index];
                            float host_result =
                                cpu_results[embedding_index][batch_index][col_index];

                            __attribute__((unused)) float diff;

                            diff = fabs(dpu_result * pow(10, 9) - host_result);
                            if (diff > CHECK_RESULT_ABSOLUTE_ROUNDING_ERROR_MARGIN)
                                printf("[%lu][%d][%d] diff: %f\tdpu_result: %f\thost_result: %f\n",
                                       embedding_index, batch_index, col_index, diff,
                                       dpu_result * pow(10, 9), host_result);

                            /* check magnitude with arbitrary threshold */
                            if (diff > CHECK_RESULT_ABSOLUTE_ROUNDING_ERROR_MARGIN)
                                valid = false;
                        }
                    }
                }
            }
#endif
        }
        double cpu_time_ms = cpu_time / multi_run;

#if (CHECK_RESULTS == 1)

        printf("dpu [ms]: %lf, cpu [ms] %lf, dpu acceleration %lf\n DPU PRATIO %f, CPU PRATIO "
               "%f, DPU "
               "OK ? %d \n",
               dpu_time_ms, cpu_time_ms, cpu_time_ms / dpu_time_ms, dpu_p_ratio / multi_run,
               cpu_p_ratio / multi_run, (int) valid);
#else
        printf("dpu [ms]: %lf\n", dpu_time_ms);
#endif
    }
}

/**
 * @brief allocate cpu result buffer
 * @param emb_info embedding info structure
 * @param i_info input info structure
 * @return buffer
 */
float ***
alloc_cpu_result_buffer(embedding_info *emb_info, input_info *i_info) {

    float ***cpu_buffer = (float ***) malloc(emb_info->nr_embedding * sizeof(float **));
    for (uint64_t k = 0; k < emb_info->nr_embedding; k++) {
        cpu_buffer[k] = (float **) malloc(i_info->nr_batches * sizeof(float *));
        for (uint64_t j = 0; j < i_info->nr_batches; j++) {
            cpu_buffer[k][j] = (float *) malloc(emb_info->nr_cols * sizeof(float));
        }
    }
    return cpu_buffer;
}

/**
 * @brief allocate cpu result buffer
 * @param buffer cpu result buffer
 * @param emb_info embedding info structure
 * @param i_info input info structure
 */
void
free_cpu_result_buffer(float ***buffer, embedding_info *emb_info, input_info *i_info) {

    for (uint64_t k = 0; k < emb_info->nr_embedding; k++) {
        for (uint64_t j = 0; j < i_info->nr_batches; j++) {
            free(buffer[k][j]);
        }
    }
    for (uint64_t k = 0; k < emb_info->nr_embedding; k++) {
        free(buffer[k]);
    }
    free(buffer);
}

/**
 * @brief allocate DPU formated result buffer
 * @param emb_info embedding info structure
 * @param i_info input info structure
 * @return buffer
 */
float **
alloc_result_buffer(embedding_info *emb_info, input_info *i_info) {
    float **result_buffer = (float **) malloc(emb_info->nr_embedding * sizeof(float *));
    for (uint64_t k = 0; k < emb_info->nr_embedding; k++) {
        result_buffer[k] =
            (float *) malloc(i_info->nr_batches * emb_info->nr_cols * sizeof(uint32_t));
    }
    return result_buffer;
}

/**
 * @brief free DPU formated result buffer
 * @param buffer cpu result buffer
 * @param nr_embedding number of embedding
 */
void
free_result_buffer(float **buffer, uint64_t nr_embedding) {

    for (uint64_t k = 0; k < nr_embedding; k++) {
        free(buffer[k]);
    }
    free(buffer);
}

/**
 * @brief alloc DPU results buffer
 * @param rank_mapping_info rank embedding mapping info
 * @param emb_info embedding info structure
 * @return buffer
 */
int32_t **
alloc_dpu_result_buffer(embedding_rank_mapping *rank_mapping_info, embedding_info *emb_info,
                        input_info *i_info) {

    uint32_t nr_dpus = rank_mapping_info->nr_dpus;
    uint64_t nr_batches = i_info->nr_batches;
    uint64_t nr_cols_per_dpu = rank_mapping_info->nr_cols_per_dpu;

    int32_t **dpu_result_buffer = (int32_t **) malloc(nr_dpus * sizeof(int32_t *));
    for (uint64_t k = 0; k < nr_dpus; k++) {
        dpu_result_buffer[k] = (int32_t *) malloc(nr_batches * nr_cols_per_dpu * sizeof(int32_t));
    }
    return dpu_result_buffer;
}

/**
 * @brief free DPU result buffer
 * @param nr_dpus total number of allocated DPUs
 * @param dpu_result_buffer buffer
 */
void
free_dpu_result_buffer(uint32_t nr_dpus, int32_t **dpu_result_buffer) {
    for (uint64_t k = 0; k < nr_dpus; k++) {
        free(dpu_result_buffer[k]);
    }
    free(dpu_result_buffer);
}
