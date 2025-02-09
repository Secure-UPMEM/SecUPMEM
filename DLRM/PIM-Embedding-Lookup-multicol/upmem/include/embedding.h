#ifndef __EMB__
#define __EMB__

#include "common.h"
#include "emb_types.h"
#include "../../PIM-common/host/include/host.h"

#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

/** @brief information about embedding input
 *  @param indices_len store dpu indices len (elem vector x batch size) for each DPU
 *  @param nr_batches input batch size
 *  @param nr_indexex nomber of indices in each input element
 */
typedef struct input_info {
    uint64_t *indices_len;
    uint64_t nr_batches;
    uint64_t nr_indexes;
} input_info;

/** @brief agregates input batch buffer structure for pipelined system
 *  @param valid validity of current batch
 *  @param indices array that stores indices [EMB_INDEX][BATCH_INDEX * INDEXES]
 *  @param offsets array that stores indices offset (pytorch EmbedingBag convention)
 *  @param input_info input info structure
 */
typedef struct input_batch {
    bool valid;
    uint32_t **indices;
    uint32_t **offsets;
    input_info *input_info;
} input_batch;

/** @brief single DPU embedding mapping information structure
 *  @param nr_cols number of collumn in the DPU
 *  @param start_col index of first column in the DPU
 *  @param embedding_index embedding index mapped in the DPU
 */
typedef struct embedding_dpu_mapping {
    uint64_t nr_cols;
    uint32_t start_col;
    uint32_t embedding_index;
} embedding_dpu_mapping;

/** @brief information about embedding configuration
 *  @param nr_embedding number of embedding
 *  @param nr_rows number of rows in the DPU
 *  @param nr_cols number of collumn in the DPU
 *  @param start_col index of first column in the DPU
 *  @param sizeT embedding data size (byte)
 */
typedef struct embedding_info {
    uint32_t nr_embedding;
    uint32_t nr_rows;
    uint32_t nr_cols;
    uint32_t sizeT;
} embedding_info;

/** @brief global ranks embedding mapping information structure
 *  @param nr_dpus total number of DPUs
 *  @param nr_ranks total number of ranks
 *  @param nr_cols_per_dpu full DPU number of column
 *  @param dpu_part_col non full DPU number of column
 *  @param rank_nr_dpus number of ranks in each DPU
 *  @param rank_start_dpus absolute index of first DPU in each rank
 *  @param rank_dpus_mapping dpu mapping matrix for each DPU of each rank
 */
typedef struct embeding_rank_mapping {
    uint32_t nr_dpus;
    uint32_t nr_ranks;
    uint64_t nr_cols_per_dpu;
    uint64_t dpu_part_col;
    uint32_t *rank_nr_dpus;
    uint32_t *rank_start_dpus;
    embedding_dpu_mapping **rank_dpus_mapping;
} embedding_rank_mapping;

embedding_rank_mapping *
get_embedding_dpu_mapping(uint64_t nr_rows, uint32_t sizeT, uint64_t nr_cols,
                          uint64_t nr_embedding);

void
alloc_dpus(uint32_t nr_dpus);

void
free_embedding_rank_mapping(embedding_rank_mapping *rank_mapping);

embedding_rank_mapping *
embedding_dpu_map(embedding_info *emb_info, input_info *i_info);

void
populate_mram(embedding_rank_mapping *rank_mapping, embedding_info *emb_info, int32_t **emb_tables);

dpu_error_t
post_process(struct dpu_set_t dpu_rank, uint64_t rank_id, void *arg);

void
lookup(uint32_t **indices, struct query_len *lengths, uint32_t **offsets, struct input_info *input_info,
       embedding_rank_mapping *rank_mapping_info, uint64_t nr_embedding, uint64_t nr_cols,
       uint64_t nr_rows, float **result_buffer, int32_t **dpu_result_buffer);

void
free_dpu_backend();

void
alloc_dpu_backend();

#endif