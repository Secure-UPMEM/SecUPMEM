#include "embedding.h"
#include "fifo.h"

#include <assert.h>
#include <dpu.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/** @brief global static fifo pool structure */
static struct FIFO_POOL *FIFO_POOL = NULL;

int32_t **
alloc_emb_tables(embedding_info *emb_info);

void
free_emb_tables(int32_t **emb_tables, embedding_info *emb_info);

void
synthetic_populate(embedding_rank_mapping *rank_mapping, embedding_info *emb_info,
                   input_info *i_info, int32_t **emb_tables);

void
cpu_embedding_lookup(uint64_t embedding_index_start, int32_t **emb_tables, uint32_t **indices,
                     uint32_t **offsets, uint64_t *indices_len, uint64_t nr_batches,
                     uint64_t nr_cols, float ***cpu_results, uint64_t nr_embedding);

void *
thread_cpu_embedding_lookup(void *argv);

void
cpu_lookup(uint64_t nr_thread, int32_t **emb_tables, uint64_t nr_embedding, uint32_t **indices,
           uint32_t **offsets, uint64_t *indices_len, uint64_t nr_batches, uint64_t nr_cols,
           float ***cpu_results);
uint32_t **
alloc_indices_buffer(uint64_t nr_embedding, uint64_t nr_batches, uint64_t indices_per_batch);

void
free_indices_buffer(uint32_t **indices, uint64_t nr_embedding);

uint32_t **
alloc_offset_buffer(uint64_t nr_embedding, uint64_t nr_batches);

void
free_offset_buffer(uint32_t **offsets, uint64_t nr_embedding);

input_info *
alloc_input_info(uint64_t nr_embedding, uint64_t nr_batches, uint64_t index_per_batch);

void
free_input_info(struct input_info *info);

embedding_info *
alloc_embedding_info(uint32_t nr_embedding, uint32_t nr_rows, uint32_t nr_cols, uint32_t sizeT);

void
free_embedding_info(embedding_info *emb_info);

void
build_synthetic_input_size(struct input_info *input_info, uint32_t **indices_per_batch,
                           uint64_t nr_embedding, uint64_t nr_batches, uint64_t nr_rows);

void
build_synthetic_input_data(uint32_t **indices, uint32_t **offsets, struct input_info *input_info,
                           uint64_t nr_embedding, uint64_t nr_batches, uint32_t **indices_per_batch,
                           uint64_t nr_rows, uint64_t nr_cols);
void
synthetic_inference(uint32_t **indices, uint32_t **offsets, input_info *input_info,
                    embedding_rank_mapping *rank_mapping_info, int32_t **emb_tables,
                    float ***cpu_results, float **result_buffer, int32_t **dpu_result_buffer,
                    embedding_info *emb_info);

float ***
alloc_cpu_result_buffer(embedding_info *emb_info, input_info *i_info);

void
free_cpu_result_buffer(float ***buffer, embedding_info *emb_info, input_info *i_info);

float **
alloc_result_buffer(embedding_info *emb_info, input_info *i_info);

void
free_result_buffer(float **buffer, uint64_t nr_embedding);

int32_t **
alloc_dpu_result_buffer(embedding_rank_mapping *rank_mapping_info, embedding_info *emb_info,
                        input_info *i_info);

void
free_dpu_result_buffer(uint32_t nr_dpus, int32_t **dpu_result_buffer);