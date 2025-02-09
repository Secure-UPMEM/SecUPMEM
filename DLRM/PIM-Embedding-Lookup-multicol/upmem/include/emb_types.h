#ifndef __EMB_TYPES_
#define __EMB_TYPES_

#include <stdbool.h>
#include <stdint.h>

#define MAX_ENC_BUFFER_SIZE MEGABYTE(MAX_ENC_BUFFER_MB)
#define MAX_CAPACITY MEGABYTE(14) // Must be a multiply of 2
#define DPUS_PER_RANK 64
#define AVAILABLE_RANKS 20
#define MAX_NR_BUFFERS 65

struct buffer_meta {
    uint64_t col_id;
    uint64_t embedding_index;
} __attribute__((packed));

struct embedding_table {
    uint64_t rank_id;
    struct dpu_set_t *rank;
    uint64_t nr_rows;
};

struct query_len {
    uint64_t indices_len;
    uint64_t nr_batches;
} __attribute__((packed));

#endif