#include <stdint.h>
#include <stdbool.h>

#define MAX_ENC_BUFFER_SIZE MEGABYTE(MAX_ENC_BUFFER_MB)
#define MAX_CAPACITY MEGABYTE(14) //Must be a multiply of 2
#define DPUS_PER_RANK 64
#define AVAILABLE_RANKS 30
#define MAX_NR_BUFFERS 65

struct buffer_meta {
    uint32_t col_id;
    uint32_t table_id;
} __attribute__((packed));

struct embedding_table {
    uint32_t rank_id;
    struct dpu_set_t *rank;
    uint64_t nr_rows;
};

struct query_len {
    uint32_t indices_len;
    uint32_t nr_batches;
}__attribute__((packed));
