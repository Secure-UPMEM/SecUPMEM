#include "embedding.h"

#include "dpu.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/** @brief DPU binary path */
#define DPU_BINARY "./build/embdpu"

/** @brief global referene to dpu_set */
struct dpu_set_t dpu_set;

/**
 * @brief free embedding rank mapping structure
 * @param rank_mapping embedding rank mapping structure
 */
void
free_embedding_rank_mapping(embedding_rank_mapping *rank_mapping) {
    for (uint32_t rank_index = 0; rank_index < rank_mapping->nr_ranks; rank_index++) {
        free(rank_mapping->rank_dpus_mapping[rank_index]);
    }
    free(rank_mapping->rank_dpus_mapping);
    free(rank_mapping->rank_nr_dpus);
    free(rank_mapping);
}

/**
 * @brief build embedding rank mapping structure
 * @param emb_info embedding info structure
 * @param i_info input info structure
 * @return embedding rank mapping structure
 */
embedding_rank_mapping *
embedding_dpu_map(embedding_info *emb_info, input_info *i_info) {

    uint64_t nr_embedding = emb_info->nr_embedding;
    uint64_t nr_rows = emb_info->nr_rows;
    uint64_t nr_cols = emb_info->nr_cols;

    uint64_t nr_dpus = 0;
    uint32_t sizeT = DATA_SIZE_BYTE;
    emb_info->sizeT = sizeT;

    embedding_rank_mapping *rank_mapping = malloc(sizeof(embedding_rank_mapping));

    uint64_t nr_cols_per_dpu;

    /* Here we are performing alignement of the minimum number of
     * columns to ensure DPU MRAM alignement and 64 bits minimum transfer
     *
     * Each DPU read of Embedding table is perfomed on a complete line
     * (eg : 2 column per DPU, datatype = int32_t ->  DPU read size = 2 * 4 = 8 Byte : OK
     *       1 column per DPU, datatype = int32_t ->  DPU read size = 1 * 4 = 4 Byte : KO
     *       2 column per DPU, datatype = int16_t ->  DPU read size = 2 * 2 = 4 Byte : KO
     *       2 column per DPU, datatype = int16_t ->  DPU read size = 2 * 2 = 4 Byte : KO
     *       4 column per DPU, datatype = int16_t ->  DPU read size = 2 * 4 = 8 Byte : KO
     *  )
     */
    uint64_t min_col_per_dpu = 1;
    while (min_col_per_dpu * sizeT % 8)
        min_col_per_dpu++;

    printf("min nr cols per dpu %lu\n", min_col_per_dpu);

    /* check if the minimum number of column fit the MRAM EMB DATA SECTION */
    assert("MRAM emb data too small" &&
           nr_rows * sizeT * min_col_per_dpu < MAX_DPU_EMB_TABLE_SIZE_BYTE);
    nr_cols_per_dpu = MAX_DPU_EMB_TABLE_SIZE_BYTE / (nr_rows * sizeT);
    if (nr_cols_per_dpu > nr_cols)
        nr_cols_per_dpu = nr_cols;

    /* align number of column to reach 8 byte allignement of DPUs column size */
    while ((nr_cols_per_dpu * sizeT) % 8)
        nr_cols_per_dpu--;

    assert(nr_cols_per_dpu > 0);

    uint64_t dpu_part_col = nr_cols % nr_cols_per_dpu;
    printf("nr cols per dpus %lu, dpu part col %lu\n", nr_cols_per_dpu, dpu_part_col);

    rank_mapping->nr_cols_per_dpu = nr_cols_per_dpu;
    rank_mapping->dpu_part_col = dpu_part_col;
    printf("MRAM_SIZE %u MAX_DPU_EMB_TABLE_SIZE_BYTE %lu nr cols per dpus %lu\n", MRAM_SIZE,
           MAX_DPU_EMB_TABLE_SIZE_BYTE, nr_cols_per_dpu);
    /* The code below is used to compute the required number of DPUs */
    {
        uint32_t dpu_total_cols = 0;
        uint32_t embedding_index = 0;
        uint32_t embedding_cur_col = 0;
        uint32_t embedding_remaining_col = nr_cols;

        while (1) {

            uint64_t dpu_nr_cols = nr_cols_per_dpu;
            if (embedding_remaining_col < dpu_nr_cols)
                dpu_nr_cols = embedding_remaining_col;
            assert(0 == (dpu_nr_cols * sizeT % 8));
            embedding_remaining_col -= dpu_nr_cols;
            embedding_cur_col += dpu_nr_cols;
            nr_dpus++;
            dpu_total_cols += dpu_nr_cols;

            if (!embedding_remaining_col) {
                embedding_remaining_col = nr_cols;
                embedding_cur_col = 0;
                embedding_index++;
            }

            if (embedding_index >= nr_embedding)
                break;
        }
    }

    printf("nr_dpus %lu\n", nr_dpus);

    DPU_ASSERT(dpu_alloc(nr_dpus, 0, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    {
        DPU_ASSERT(dpu_get_nr_ranks(dpu_set, &(rank_mapping->nr_ranks)));
        {
            uint32_t rank_index = 0;
            rank_mapping->rank_nr_dpus = malloc(rank_mapping->nr_ranks * sizeof(uint32_t));
            rank_mapping->rank_start_dpus = malloc(rank_mapping->nr_ranks * sizeof(uint32_t));

            struct dpu_set_t rank;
            uint32_t total_dpus = 0;
            DPU_RANK_FOREACH(dpu_set, rank, rank_index) {
                DPU_ASSERT(dpu_get_nr_dpus(rank, &(rank_mapping->rank_nr_dpus[rank_index])));
                rank_mapping->rank_start_dpus[rank_index] = total_dpus;
                total_dpus += rank_mapping->rank_nr_dpus[rank_index];
            }
        }

        rank_mapping->rank_dpus_mapping =
            malloc(rank_mapping->nr_ranks * sizeof(embedding_dpu_mapping *));
        for (uint64_t rank_index = 0; rank_index < rank_mapping->nr_ranks; rank_index++)
            rank_mapping->rank_dpus_mapping[rank_index] =
                malloc(rank_mapping->rank_nr_dpus[rank_index] * sizeof(embedding_dpu_mapping));

        uint32_t rank_index = 0;
        uint32_t rank_dpu_index = 0;
        uint32_t dpu_total_cols = 0;
        uint32_t rank_remaining_dpus = rank_mapping->rank_nr_dpus[rank_index];
        uint32_t embedding_index = 0;
        uint32_t embedding_cur_col = 0;
        uint32_t embedding_remaining_col = nr_cols;

        while (1) {
            assert(rank_index < rank_mapping->nr_ranks);
            assert(rank_dpu_index < rank_mapping->rank_nr_dpus[rank_index]);

            uint64_t dpu_nr_cols = nr_cols_per_dpu;
            if (embedding_remaining_col < dpu_nr_cols)
                dpu_nr_cols = embedding_remaining_col;

            assert(0 == (dpu_nr_cols * sizeT % 8));

            rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].nr_cols = dpu_nr_cols;
            rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].start_col =
                embedding_cur_col;
            rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index =
                embedding_index;

            embedding_remaining_col -= dpu_nr_cols;
            embedding_cur_col += dpu_nr_cols;
            rank_dpu_index++;
            rank_remaining_dpus--;
            dpu_total_cols += dpu_nr_cols;

            if (!embedding_remaining_col) {
                embedding_remaining_col = nr_cols;
                embedding_cur_col = 0;
                embedding_index++;
            }

            if (!rank_remaining_dpus) {
                rank_index++;
                rank_dpu_index = 0;
                rank_remaining_dpus = rank_mapping->rank_nr_dpus[rank_index];
            }

            if (embedding_index >= nr_embedding)
                break;
        }
    }
#define DBG 0
#if (DBG == 1)
    for (uint32_t rank_index = 0; rank_index < rank_mapping->nr_ranks; rank_index++) {
        for (uint32_t rank_dpu_index = 0; rank_dpu_index < rank_mapping->rank_nr_dpus[rank_index];
             rank_dpu_index++) {
            printf("rank %u dpu %u emb index %u start col %u nr col %lu\n", rank_index,
                   rank_dpu_index,
                   rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index,
                   rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].start_col,
                   rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].nr_cols);
        }
    }
#endif
#undef DBG
    rank_mapping->nr_dpus = nr_dpus;
    return rank_mapping;
}

/** @brief transfer embedding tables to DPUs MRAM
 *  @param rank_mapping embedding rank mapping structure
 *  @param emb_info embedding info structure
 *  @param embedding_tables embedding tables buffer
 */
void
populate_mram(embedding_rank_mapping *rank_mapping, embedding_info *emb_info,
              int32_t **emb_tables) {

    uint64_t sizeT = emb_info->sizeT;
    uint32_t nr_dpus = rank_mapping->nr_dpus;
    uint32_t nr_cols = emb_info->nr_cols;
    uint32_t nr_rows = emb_info->nr_rows;
    uint32_t nr_cols_per_dpu = rank_mapping->nr_cols_per_dpu;
    uint32_t dpu_part_col = rank_mapping->dpu_part_col;
    /* allocates ant creates transpose embeding matrix of parameters */
    int32_t **buffer_data;
    buffer_data = (int32_t **) (malloc(nr_dpus * sizeof(int32_t *)));
    for (uint64_t dpu_index = 0; dpu_index < nr_dpus; dpu_index++)
        buffer_data[dpu_index] = (int32_t *) (malloc(nr_rows * nr_cols_per_dpu * sizeof(int32_t)));

    struct dpu_set_t dpu;
    struct dpu_set_t rank;
    uint32_t dpu_index = 0;
    uint64_t nr_dpus_ = 0;
    uint32_t rank_dpu_index;
    uint32_t rank_index;

    DPU_RANK_FOREACH(dpu_set, rank, rank_index) {
        DPU_FOREACH(rank, dpu, rank_dpu_index) {
            uint64_t emb_index =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index;
            uint64_t start_col =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].start_col;
            if (rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].nr_cols ==
                nr_cols_per_dpu) {
                for (uint64_t row_index = 0; row_index < nr_rows; row_index++) {
                    for (uint64_t col_index = start_col, dpu_col_index = 0;
                         col_index < nr_cols_per_dpu + start_col; col_index++, dpu_col_index++) {
                        buffer_data[dpu_index][row_index * nr_cols_per_dpu + dpu_col_index] =
                            emb_tables[emb_index][row_index * nr_cols + col_index];
                    }
                }
            } else if (rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].nr_cols ==
                       dpu_part_col) {
                for (uint64_t row_index = 0; row_index < nr_rows; row_index++) {
                    for (uint64_t col_index = start_col, dpu_col_index = 0;
                         col_index < dpu_part_col + start_col; col_index++, dpu_col_index++) {
                        buffer_data[dpu_index][row_index * dpu_part_col + dpu_col_index] =
                            emb_tables[emb_index][row_index * nr_cols + col_index];
                    }
                }
            } else {
                assert(false && "exception\n");
            }

            DPU_ASSERT(dpu_prepare_xfer(dpu, buffer_data[dpu_index]));
            dpu_index++;
        }
    }

    printf("start xfer %lu part dpus with size %lu nr cols %u\n", nr_dpus_,
           nr_rows * sizeT * dpu_part_col, dpu_part_col);
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "emb_data", 0,
                             nr_rows * sizeT * nr_cols_per_dpu, DPU_XFER_DEFAULT));

    for (uint64_t dpu_index = 0; dpu_index < nr_dpus; dpu_index++)
        free(buffer_data[dpu_index]);
    free(buffer_data);

    DPU_RANK_FOREACH(dpu_set, rank, rank_index) {
        DPU_FOREACH(rank, dpu, rank_dpu_index) {
            DPU_ASSERT(dpu_prepare_xfer(
                dpu, &(rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].nr_cols)));
        }
    }

    DPU_ASSERT(
        dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "nr_cols", 0, sizeof(uint64_t), DPU_XFER_DEFAULT));

    return;
}
struct callback_input {
    float **result_buffer;
    uint64_t nr_batches;
    int32_t **dpu_results_buffer;
    uint64_t nr_cols;
    uint64_t nr_rows;
    uint64_t nr_embedding;
    embedding_rank_mapping *rank_mapping_info;
};

struct callback_input *callback_data = NULL;

/** @brief rank callback for DPU results post processing
 *  @param rank dpu_set rank pointer
 *  @param rank_index index of the rank
 *  @param cb_args thread function args
 */
dpu_error_t
gather_rank_embedding_results(struct dpu_set_t rank, uint32_t rank_index, void *cb_arg) {
    __attribute__((unused)) struct dpu_set_t dpu;

    struct callback_input *input = (struct callback_input *) cb_arg;
    int32_t **dpu_results_buffer = input->dpu_results_buffer;
    float **result_buffer = input->result_buffer;
    uint64_t nr_batches = input->nr_batches;
    uint64_t nr_cols = input->nr_cols;
    embedding_rank_mapping *rank_mapping = input->rank_mapping_info;

    uint32_t rank_start_dpus = rank_mapping->rank_start_dpus[rank_index];

    uint32_t rank_dpu_index;
    uint32_t dpu_index;

    /* gather plain column DPUs sub-set */
    DPU_FOREACH(rank, dpu, rank_dpu_index) {
        dpu_index = rank_dpu_index + rank_start_dpus;
        if (rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].nr_cols ==
            rank_mapping->nr_cols_per_dpu) {
            uint64_t emb_index =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index;
            uint64_t start_col =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].start_col;
            for (uint64_t batch_index = 0; batch_index < nr_batches; batch_index++) {
                for (uint64_t col_index = 0; col_index < rank_mapping->nr_cols_per_dpu;
                     col_index++) {
                    int32_t dpu_result =
                        dpu_results_buffer[dpu_index]
                                          [(batch_index * rank_mapping->nr_cols_per_dpu) +
                                           col_index];
                    result_buffer[emb_index][start_col + col_index + (batch_index * nr_cols)] =
                        (float) (dpu_result) *pow(10, -9);
                }
            }
        }
    }

    /* gather part column DPUs sub-set */
    DPU_FOREACH(rank, dpu, rank_dpu_index) {
        dpu_index = rank_dpu_index + rank_start_dpus;
        if (rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].nr_cols ==
            rank_mapping->dpu_part_col) {
            uint64_t emb_index =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index;
            uint64_t start_col =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].start_col;
            for (uint64_t batch_index = 0; batch_index < nr_batches; batch_index++) {
                for (uint64_t col_index = 0; col_index < rank_mapping->dpu_part_col; col_index++) {
                    int32_t dpu_result =
                        dpu_results_buffer[dpu_index]
                                          [(batch_index * rank_mapping->dpu_part_col) + col_index];
                    result_buffer[emb_index][start_col + col_index + (batch_index * nr_cols)] =
                        (float) (dpu_result) *pow(10, -9);
                }
            }
        }
    }

    return DPU_OK;
}

/** @brief perform DPU lookup operation
 *  @param indices array that stores indices [EMB_INDEX][BATCH_INDEX * INDEXES]
 *  @param offsets array that stores indices offset (pytorch EmbedingBag convention)
 *  @param input_info input info structure
 *  @param rank_mapping embedding rank mapping structure
 *  @param nr_embedding number of embedding
 *  @param nr_cols number of embedding column
 *  @param n_rows number of embedding rows
 *  @param result_buffer DPU formated result buffer
 *  @param dpu_result_buffer dpu_result_buffer
 */
void
lookup(uint32_t **indices, struct query_len *lengths, uint32_t **offsets, input_info *input_info,
       embedding_rank_mapping *rank_mapping, uint64_t nr_embedding, uint64_t nr_cols,
       uint64_t nr_rows, float **result_buffer, int32_t **dpu_result_buffer) {

    uint32_t rank_index;
    uint32_t rank_dpu_index;
    struct dpu_set_t dpu;
    struct dpu_set_t rank;

    uint64_t sizeT = sizeof(int32_t);

    uint32_t max_nr_batches = input_info->nr_batches;
    uint32_t max_indices_len = 0;

    for (uint64_t emb_index = 0; emb_index < nr_embedding; emb_index++) {
        if (max_indices_len < input_info->indices_len[emb_index])
            max_indices_len = input_info->indices_len[emb_index];
    }

    DPU_RANK_FOREACH(dpu_set, rank, rank_index) {
        DPU_FOREACH(rank, dpu, rank_dpu_index) {
            uint64_t emb_index =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index;
            assert(emb_index < MAX_NR_EMBEDDING);
        }
    }

    DPU_RANK_FOREACH(dpu_set, rank, rank_index) {
        DPU_FOREACH(rank, dpu, rank_dpu_index) {
            uint64_t emb_index =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index;
            DPU_ASSERT(dpu_prepare_xfer(dpu, indices[emb_index]));
        }
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_indices", 0,
                             ALIGN(max_indices_len * sizeof(uint32_t), 8), DPU_XFER_ASYNC));

    DPU_RANK_FOREACH(dpu_set, rank, rank_index) {
        DPU_FOREACH(rank, dpu, rank_dpu_index) {
            uint64_t emb_index =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index;
            DPU_ASSERT(dpu_prepare_xfer(dpu, offsets[emb_index]));
        }
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_offsets", 0,
                             ALIGN(max_nr_batches * sizeof(uint32_t), 8), DPU_XFER_ASYNC));

    DPU_RANK_FOREACH(dpu_set, rank, rank_index) {
        DPU_FOREACH(rank, dpu, rank_dpu_index) {
            uint64_t emb_index =
                rank_mapping->rank_dpus_mapping[rank_index][rank_dpu_index].embedding_index;
            lengths[emb_index].indices_len = input_info->indices_len[emb_index];
            lengths[emb_index].nr_batches = input_info->nr_batches;
            DPU_ASSERT(dpu_prepare_xfer(dpu, &lengths[emb_index]));
        }
    }

    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "input_lengths", 0, sizeof(struct query_len),
                             DPU_XFER_ASYNC));
    DPU_ASSERT(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
#if (PERFCOUNT == 1)
    {
        uint32_t dpu_index;
        aPU_ASSERT(dpu_sync(dpu_set));
        DPU_FOREACH(dpu_set, dpu, dpu_index) {
            DPU_ASSERT(dpu_log_read(dpu, stdout));
            break;
        }
    }
#endif

#if (DPUDBG == 1)
    {
        uint32_t dpu_index;
        DPU_ASSERT(dpu_sync(dpu_set));
        DPU_FOREACH(dpu_set, dpu, dpu_index) {
            DPU_ASSERT(dpu_log_read(dpu, stdout));
        }
    }
#endif

    uint32_t dpu_index;
    DPU_FOREACH(dpu_set, dpu, dpu_index) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, dpu_result_buffer[dpu_index]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "results", 0,
                             ALIGN(sizeT * max_nr_batches * rank_mapping->nr_cols_per_dpu, 8),
                             DPU_XFER_ASYNC));

    callback_data->nr_cols = nr_cols;
    callback_data->nr_rows = nr_rows;
    callback_data->result_buffer = result_buffer;
    callback_data->nr_batches = input_info->nr_batches;
    callback_data->dpu_results_buffer = dpu_result_buffer;
    callback_data->rank_mapping_info = rank_mapping;
    DPU_ASSERT(
        dpu_callback(dpu_set, gather_rank_embedding_results, callback_data, DPU_CALLBACK_DEFAULT));
}

/** @brief allocate DPU backend */
void
alloc_dpu_backend() {
    assert(callback_data == NULL);
    callback_data = malloc(sizeof(struct callback_input));
}

/** @brief free DPU backend */
void
free_dpu_backend() {
    free(callback_data);
}
