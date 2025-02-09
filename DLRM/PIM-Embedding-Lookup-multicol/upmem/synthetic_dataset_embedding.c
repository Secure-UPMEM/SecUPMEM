#include "embedding.h"
#include "emblib.h"
#include "fifo.h"

#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define THREAD_POOL_NR_THREAD 1
struct THREAD_POOL {
    pthread_t th[THREAD_POOL_NR_THREAD];
    struct pipeline_args *pargs;
} THREAD_POOL;

#define STAGE_0_DEPTH 2
#define STAGE_1_DEPTH 2

struct FIFO_POOL {
    FIFO stage_0;
    FIFO stage_1;
};
/**
 * @brief alloc pipeline fifo pool structure
 * @param emb_info embedding info structure
 * @param i_info input info structure
 * @return fifo pool structure
 */
struct FIFO_POOL *
alloc_fifo_pool(embedding_info *emb_info, input_info *i_info) {

    struct FIFO_POOL *this = malloc(sizeof(struct FIFO_POOL));

    {
        printf("alloc FIFO [build_synthetic_input_data->inference], DEPTH(%u)\n", STAGE_0_DEPTH);
        input_batch *batch = malloc(sizeof(input_batch) * STAGE_0_DEPTH);
        for (uint64_t i = 0; i < STAGE_0_DEPTH; i++) {
            uint32_t **indices = alloc_indices_buffer(emb_info->nr_embedding, i_info->nr_batches,
                                                      INDICES_PER_LOOKUP);
            uint32_t **offsets = alloc_offset_buffer(emb_info->nr_embedding, i_info->nr_batches);
            batch[i].indices = indices;
            batch[i].offsets = offsets;
            batch[i].input_info = i_info;
        }
        FIFO_INIT(&(this->stage_0), (void *) (batch), STAGE_0_DEPTH, sizeof(input_batch), 1, 1);
    }

    return this;
}

/**
 * @brief free pipeline fifo pool structure
 * @param this fifo pool structure
 * @param emb_info embedding info structure
 */
void
free_fifo_pool(struct FIFO_POOL *this, embedding_info *emb_info) {

    {
        FIFO *fifo = &(this->stage_0);
        printf("free FIFO [build_synthetic_input_data->inference], DEPTH(%lu)\n", fifo->depth);
        input_batch *batch = (input_batch *) (fifo->items[0]);
        for (uint64_t i = 0; i < STAGE_0_DEPTH; i++) {
            free_indices_buffer(batch[i].indices, emb_info->nr_embedding);
            free_offset_buffer(batch[i].offsets, emb_info->nr_embedding);
        }
        free(batch);
        FIFO_FREE(fifo);
    }
}

struct pipeline_args {
    embedding_rank_mapping *rank_mapping_info;
    embedding_info *emb_info;
    input_info *i_info;
};
/**
 * @brief thread function for input indices generation
 * @param argv thread args
 */
void *
thread_build_sythetic_data(void *argv) {

    struct pipeline_args *args = (struct pipeline_args *) argv;

    embedding_info *emb_info = args->emb_info;
    input_info *i_info = args->i_info;

    uint64_t nr_rows = emb_info->nr_rows;
    uint64_t nr_cols = emb_info->nr_cols;
    uint64_t nr_embedding = emb_info->nr_embedding;
    uint64_t nr_batches = i_info->nr_batches;

    FIFO *OUTPUT_FIFO = &(FIFO_POOL->stage_0);
    uint64_t total_batch = 0;

    uint32_t **indices_per_batch;
    indices_per_batch = malloc(NR_EMBEDDING * sizeof(uint32_t *));
    printf("max nr embedding %u\n", NR_EMBEDDING);
    for (uint64_t batch_index = 0; batch_index < NR_EMBEDDING; batch_index++)
        indices_per_batch[batch_index] = malloc(BATCH_SIZE * sizeof(uint32_t));

    while (1) {
        input_batch *batch = FIFO_PUSH_RESERVE(input_batch, *OUTPUT_FIFO);
        batch->valid = 1;

        if (!(total_batch++ < NR_RUN)) {
            batch->valid = 0;
            FIFO_PUSH_RELEASE(*OUTPUT_FIFO);
            break;
        }

        /* creates synthetic input batch of data */
        build_synthetic_input_size(batch->input_info, indices_per_batch, nr_embedding, nr_batches,
                                   nr_rows);
        /* creates synthetic input batch of data */
        build_synthetic_input_data(batch->indices, batch->offsets, batch->input_info, nr_embedding,
                                   nr_batches, indices_per_batch, nr_rows, nr_cols);

        /* release input FIFO */
        FIFO_PUSH_RELEASE(*OUTPUT_FIFO);
    }

    for (uint64_t batch_index = 0; batch_index < NR_EMBEDDING; batch_index++)
        free(indices_per_batch[batch_index]);
    free(indices_per_batch);

    /* thread exit */
    pthread_exit(NULL);
    return NULL;
}

/** TBC
 * @brief thread function for DPU output results merging
 * @param argv thread args
 */
void *
thread_mege_results(void *argv) {
    /* thread exit */
    return NULL;
}

/**
 * @brief initialize thread pool structure
 * @param this fifo pool structure
 * @param rank_mapping_info rank embedding mapping info
 * @param emb_info embedding info structure
 * @param i_info input info structure
 */
void
INIT_THREAD_POOL(struct THREAD_POOL *this, embedding_rank_mapping *rank_mapping_info,
                 embedding_info *emb_info, input_info *i_info) {
    this->pargs = (struct pipeline_args *) malloc(sizeof(struct pipeline_args));
    this->pargs->rank_mapping_info = rank_mapping_info;
    this->pargs->emb_info = emb_info;
    this->pargs->i_info = i_info;
    /* index on current thread to of THREAD_POOL structure */
    uint64_t thread_pool_index = 0;
    pthread_create(&(this->th[thread_pool_index++]), NULL, thread_build_sythetic_data,
                   (void *) this->pargs);
}

/**
 * @brief join all thread of thread pool
 * @param this thread pool structure
 */
void
JOIN_THREAD_POOL(struct THREAD_POOL *this) {
    for (uint64_t i = 0; i < THREAD_POOL_NR_THREAD; i++)
        pthread_join(this->th[i], NULL);
    free(this->pargs);
}

/** @brief perform DPU/CPU benchmark of embedding tables */
int
main() {

    uint64_t nr_embedding = NR_EMBEDDING;
    uint64_t nr_batches = BATCH_SIZE;
    uint64_t indices_per_lookup = INDICES_PER_LOOKUP;
    uint64_t nr_cols = EMBEDDING_DIM;
    uint64_t nr_rows = EMBEDDING_DEPTH;

    input_info *i_info = alloc_input_info(nr_embedding, nr_batches, indices_per_lookup);
    embedding_info *emb_info =
        alloc_embedding_info(nr_embedding, nr_rows, nr_cols, sizeof(int32_t));

    FIFO_POOL = alloc_fifo_pool(emb_info, i_info);
    alloc_dpu_backend();

    /* alloc final results buffer */
    printf("map embeddings on DPUs\n");
    embedding_rank_mapping *rank_mapping = embedding_dpu_map(emb_info, i_info);

    printf("nr cols per dpu %lu\n", rank_mapping->nr_cols_per_dpu);

    uint32_t nr_dpus = rank_mapping->nr_dpus;
    printf("alloc dpus %u\n", nr_dpus);
    int32_t **emb_tables = alloc_emb_tables(emb_info);

    float **result_buffer = alloc_result_buffer(emb_info, i_info);
    float ***cpu_results = alloc_cpu_result_buffer(emb_info, i_info);
    int32_t **dpu_result_buffer = alloc_dpu_result_buffer(rank_mapping, emb_info, i_info);

    /* creates synthetic embedding parametes and transfet it to DPU MRAM */
    synthetic_populate(rank_mapping, emb_info, i_info, emb_tables);

    INIT_THREAD_POOL(&THREAD_POOL, rank_mapping, emb_info, i_info);

    FIFO *INPUT_FIFO = &(FIFO_POOL->stage_0);
    printf("start inference\n");
    while (1) {

        /* get one mapped read batch */
        input_batch *batch = FIFO_POP_RESERVE(input_batch, *INPUT_FIFO);

        if (!batch->valid) {
            FIFO_POP_RELEASE(*INPUT_FIFO);
            break;
        }

        /* perform inference */
        // printf("hello\n");
        synthetic_inference(batch->indices, batch->offsets, batch->input_info, rank_mapping,
                            emb_tables, cpu_results, result_buffer, dpu_result_buffer, emb_info);
        // printf("bye\n");
        FIFO_POP_RELEASE(*INPUT_FIFO);
    }
    JOIN_THREAD_POOL(&THREAD_POOL);

    free_result_buffer(result_buffer, emb_info->nr_embedding);
    free_dpu_result_buffer(nr_dpus, dpu_result_buffer);
    free_cpu_result_buffer(cpu_results, emb_info, i_info);
    free_emb_tables(emb_tables, emb_info);

    free_fifo_pool(FIFO_POOL, emb_info);
    free_dpu_backend();
    free_embedding_info(emb_info);
    free_input_info(i_info);
}
