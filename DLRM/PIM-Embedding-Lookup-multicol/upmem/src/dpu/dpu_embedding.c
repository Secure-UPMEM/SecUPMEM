// To build the code: dpu-upmem-dpurte-clang -o toy_dpu toy_dpu.c
#include "common.h"
#include "../../PIM-common/common/include/common.h"
#include "emb_types.h"

#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>

__host uint64_t nr_cols;
__mram_noinit struct query_len input_lengths;
__mram_noinit int32_t emb_data[MAX_DPU_EMB_TABLE_SIZE];
__mram_noinit uint32_t input_indices[MAX_INDICES_PER_LOOKUP * MAX_BATCH_SIZE];
__mram_noinit uint32_t input_offsets[MAX_BATCH_SIZE];
__mram_noinit int32_t results[MAX_BATCH_SIZE * MAX_EMBEDDING_DIM];
// #define PERFCOUNT
#ifdef PERFCOUNT
__host uint32_t counter_all, counter_init;
#endif

BARRIER_INIT(my_barrier, NR_TASKLETS);

uint64_t indices_len;
uint64_t nr_batches;
__dma_aligned int32_t tmp_emb_data[NR_TASKLETS][MAX_EMBEDDING_DIM];
__dma_aligned struct query_len lengths;
__dma_aligned uint32_t indices[MAX_INDICES_PER_LOOKUP * MAX_BATCH_SIZE];
__dma_aligned uint32_t offsets[MAX_BATCH_SIZE];
__dma_aligned int32_t tmp_results[NR_TASKLETS][MAX_EMBEDDING_DIM];

int
main() {

    if (me() == 0) {

#if (DPUDBG == 1)
        printf("nr cols %lu\n", nr_cols);
        for (uint64_t r_index = 0; r_index < 100; r_index++) {
            mram_read(&emb_data[r_index * nr_cols], tmp_emb_data[0],
                      ALIGN(nr_cols * sizeof(int32_t), 8));
            for (uint64_t col_index = 0; col_index < nr_cols; col_index++) {
                printf("d %d, ", tmp_emb_data[0][col_index]);
            }
            printf("\n");
        }
        return 0;
#endif

#ifdef PERFCOUNT
        perfcounter_config(COUNT_INSTRUCTIONS, true);
#endif

        /* load all indices */
        mram_read(&input_lengths, &lengths, ALIGN(sizeof(struct query_len), 8));
        indices_len = lengths.indices_len;
        nr_batches = lengths.nr_batches;
        uint64_t copied_indices = 0;
        while (copied_indices < indices_len) {
            mram_read(&input_indices[copied_indices], &indices[copied_indices],
                      ALIGN(MIN(2048, (indices_len - copied_indices) * sizeof(uint32_t)), 8));
            copied_indices += 2048 / sizeof(uint32_t);
        }

        mram_read(input_offsets, offsets, ALIGN(nr_batches * sizeof(uint32_t), 8));

#ifdef PERFCOUNT
        counter_init = perfcounter_get();
#endif
    }
    barrier_wait(&my_barrier);

    uint64_t batch_lot_size = 2048 / sizeof(*results);
    uint64_t nr_batch_lots = (nr_batches / batch_lot_size) + 1;
    uint64_t batch_end = 0;

    uint32_t _me = me();

    /* for each element of batch lot */
    for (uint64_t batch_lot = 0; batch_lot < nr_batch_lots; batch_lot++) {
        uint64_t batch_start = batch_end;
        batch_end = MIN(batch_start + batch_lot_size, nr_batches);
        /* for each element of batch : one indice vector by tasklet */
        for (uint64_t i = _me + batch_start; i < batch_end; i += NR_TASKLETS) {
            /* reset results */
            for (uint64_t col_index = 0; col_index < nr_cols; col_index++)
                tmp_results[_me][col_index] = 0;

            uint64_t upper_bound = i == nr_batches - 1 ? indices_len : offsets[i + 1];

            /* for each indice of indice vector */
            for (uint64_t indices_ptr = offsets[i]; indices_ptr < upper_bound; indices_ptr++) {
                uint64_t ind = indices[indices_ptr];
                /* load current table row */
                mram_read(&emb_data[ind * nr_cols], tmp_emb_data[_me],
                          ALIGN(nr_cols * sizeof(int32_t), 8));
                /* for each column : rowise accumulate each element of current row indice */
                for (uint64_t col_index = 0; col_index < nr_cols; col_index++) {
                    tmp_results[_me][col_index] += tmp_emb_data[_me][col_index];
                }
            }
            /* store accumulated row */
            mram_write(tmp_results[_me], &results[i * nr_cols],
                       ALIGN(nr_cols * sizeof(int32_t), 8));
        }
    }
#if (DPUDBG == 1)
    if (me() == 0) {
        for (uint64_t n_index = 0; n_index < nr_batches; n_index++) {
            mram_read(&results[n_index * nr_cols], tmp_results[me()],
                      ALIGN(nr_cols * sizeof(int32_t), 8));
            for (uint64_t col_index = 0; col_index < nr_cols; col_index++) {
                tmp_results[n_index][col_index];

                printf("batch %lu col ind %lu res %d me() %u\n", n_index, col_index,
                       tmp_results[n_index][col_index], me());
            }
        }
    }
#endif
#ifdef PERFCOUNT
    counter_all = perfcounter_get();
#endif

#ifdef PERFCOUNT
    printf("counter all %u \n", counter_all);
    printf("counter init %u \n", counter_init);
#endif

    return 0;
}
