/*
 * Compute gradient of MSE loss function with multiple tasklet ver1.1
 * sigmoid done by Taylor Series 
 * 
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>

#include "../support/common.h"

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;  
__host uint8_t b1[NR_TASKLETS*MAX_ROWS];
__host uint8_t b2[NR_TASKLETS*MAX_ROWS];
//new added
__host uint32_t op_mode;
__host T dot_product_t[NR_TASKLETS*MAX_ROWS];
__host T dot_product_temp[NR_TASKLETS][MAX_ROWS];
// __host T sigmoid_tmp[NR_TASKLETS][MAX_ROWS];
__host T sigmoid_tmp[NR_TASKLETS*MAX_ROWS];
//end
__mram_noinit T DPU_RESULTS[MAX_ROWS]; // partial gradient in each DPU, max number of rows = 16 
// __mram_noinit T DPU_PRODUCT[MAX_ROWS*NR_TASKLETS];


__dma_aligned T gradient_tmp[MAX_ROWS*NR_TASKLETS]; // tasklet major storage 
// __dma_aligned T dot_product_t[MAX_ROWS*NR_TASKLETS];
// __dma_aligned T sigmoid_tmp[MAX_ROWS];

// Dot product 
static T dot_product(T *bufferX, T *bufferW, uint32_t length) {
    T result = 0; 
    for (unsigned int i = 0; i < length; i++) {
        result += bufferX[i] * bufferW[i]; 
    }
    return result; 
}

# ifdef FLOAT 
// Sigmoid function by Taylor serious 
static float sigmoid_dpu(T x){
    if(x >= 15)
        return (T) 1; 
    else if (x <= -15) 
        return (T) 0; 
    else if (x == 0.0)
        return 0.5; 

    float sum = 1.0;
    float temp = 1.0; 
    // iter 100 times 
    for(uint32_t i = 1; i < 101; ++i){
        temp = temp * (-x) / i;
        sum = sum + temp; 
    }
    // printf("exp: %f\n", sum);
    return (1.0 / (1.0 + sum)); 
} 

# else
// Sigmoid function by Taylor serious, fixed-point arithmetic 
static T sigmoid_dpu_fp(float x){
    if(x >= 15.0)
        return (T) (1<<SHIFT_AMOUNT); 
    else if (x <= -15.0) 
        return (T) 0; 
    else if (x == 0.0)
        return (1<<(SHIFT_AMOUNT-1)); 

    float sum = 1.0;
    float temp = 1.0; 
    // iter 100 times 
    for(uint32_t i = 1; i < 101; ++i){
        temp = temp * (-x) / i;
        sum = sum + temp; 
    }
    // printf("exp: %f\n", sum);
    return ((1<<SHIFT_AMOUNT) / (1.0 + sum)); 
}

// TODO: sogmoid function by look-up-table for int 

# endif 

// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// main
int main() {
    // printf("first load");
    unsigned int tasklet_id = me();
    if (tasklet_id == 0){ // Initialize once the cycle counter
        mem_reset(); // Reset the heap
    }
    
    barrier_wait(&my_barrier); // Barrier

    //added
    uint32_t mode = op_mode;
    //end

    uint32_t n_size = DPU_INPUT_ARGUMENTS.n_size;
    uint32_t n_size_pad = DPU_INPUT_ARGUMENTS.n_size_pad;
    uint32_t nr_rows = DPU_INPUT_ARGUMENTS.nr_rows;
    uint32_t max_rows = DPU_INPUT_ARGUMENTS.max_rows;

    // arguments for each tasklet 
    uint32_t rows_per_tasklet = DPU_INPUT_ARGUMENTS.rows_per_tasklet[tasklet_id]; 
    uint32_t start_row = DPU_INPUT_ARGUMENTS.start_row[tasklet_id];

    // Clear global arrays in WRAM 
    uint32_t tasklet_offset = tasklet_id * n_size_pad; 
    for (uint32_t each_attribute = 0; each_attribute < n_size_pad; each_attribute++) {
        gradient_tmp[tasklet_offset + each_attribute] = 0; 
    } 

    // Address of the current row in MRAM
    uint32_t n_size_byte = n_size << MUL;//* sizeof(T);
    uint32_t n_size_pad_byte = n_size_pad << MUL;//* sizeof(T); 

    uint32_t mram_base_addr_X = (uint32_t) (DPU_MRAM_HEAP_POINTER + start_row * n_size_byte);
    uint32_t mram_base_addr_Y = (uint32_t) (DPU_MRAM_HEAP_POINTER + max_rows * n_size_pad_byte + (start_row << MUL)); 
    uint32_t mram_base_addr_W = (uint32_t) (DPU_MRAM_HEAP_POINTER + max_rows * n_size_pad_byte + (max_rows << MUL)); 
    
    uint32_t mram_temp_addr_X = mram_base_addr_X;
    uint32_t mram_temp_addr_Y = mram_base_addr_Y;

    // Inititalize a local cache to store the MRAM block 
    T *cache_X = (T *) mem_alloc(BLOCK_SIZE); 
    T *cache_Y = (T *) mem_alloc(BLOCK_SIZE); 
    T *cache_W = (T *) mem_alloc(n_size_pad_byte); 

    // read W from MRAM 
    mram_read((__mram_ptr void const*) (mram_base_addr_W), cache_W, n_size_pad_byte); 

    // Iterate over nr_rows
    uint32_t rows_per_cache = BLOCK_SIZE / n_size_byte; 
    # if PRINT 
    if(tasklet_id == NR_TASKLETS-1) {
        printf("tasklet: %d, n_size: %d, n_size_pad: %d, row/tasklet: %d, nr_rows: %d, max_rows: %d\n", \
            tasklet_id, n_size, n_size_pad, rows_per_tasklet, nr_rows, max_rows); 
        printf("cache size: %d, rows_per_cache: %d\n", BLOCK_SIZE, rows_per_cache); 
    }
    # endif
    // printf("pre pre pre");
    
    for (unsigned int row_index = 0; row_index < rows_per_tasklet;) {
        mram_temp_addr_X = mram_base_addr_X + row_index * n_size_byte; 
        mram_temp_addr_Y = mram_base_addr_Y + (row_index << MUL); 

        // read X and Y from MRAM 
        mram_read((__mram_ptr void const*) (mram_temp_addr_X), cache_X, BLOCK_SIZE); 
        mram_read((__mram_ptr void const*) (mram_temp_addr_Y), cache_Y, BLOCK_SIZE); 

        // Iterate over cache 
        uint32_t x_index = 0; 
        // printf("pre pre");
        for(unsigned int y_index = 0; (y_index<rows_per_cache) && (row_index<rows_per_tasklet); y_index++, \
            row_index++){ 
            if(row_index+start_row >= nr_rows){
                row_index = rows_per_tasklet; 
                break; 
            }

            // compute dot product
            // printf("row per tasklet: %d  ", rows_per_tasklet);
            // printf("total:%d  ",tasklet_id*rows_per_tasklet+row_index);
            // printf("tasklet:%d  ",tasklet_id);
            // printf("row:%d \n",row_index);
            // printf("pre check %d",op_mode);
            if( op_mode == 0){
                
                dot_product_temp[tasklet_id][row_index] = dot_product(cache_X + x_index, cache_W, n_size); 
                // sigmoid_tmp[tasklet_id][row_index]=dot_product_temp[tasklet_id][row_index];
            }
            //newly comment added
                // # ifdef FLOAT
                // T sigmoid = sigmoid_dpu(dot_product_t); 
                // # else 
                // T sigmoid = sigmoid_dpu_fp((float) (dot_product_t>>SHIFT_AMOUNT) / (SHIFT_MASK+1)); 
                // # endif

            //end

            else if(op_mode == 1){
                // if((~b1[(row_index*rows_per_cache)+y_index] & b2[(row_index*rows_per_cache)+y_index]) == 1) sigmoid_tmp[tasklet_id][row_index]= dot_product_temp[tasklet_id][row_index];
                // else if(b2[(row_index*rows_per_cache)+y_index] == 1) sigmoid_tmp[tasklet_id][row_index]= 0;
                // else sigmoid_tmp[tasklet_id][row_index]= 1;
                // printf("%d\n",  sigmoid_tmp[tasklet_id][row_index] );
                // compute gradient 
                // TODO: unroll the loop 
                    for (unsigned int l = 0; l < n_size; ++l) {
                    #ifdef FLOAT 
                    gradient_tmp[tasklet_offset + l] += cache_X[x_index + l] * (sigmoid_tmp[row_index*rows_per_tasklet+row_index]- cache_Y[y_index]); 

                    #else // int, fixed-pointed  
                    gradient_tmp[tasklet_offset + l] += cache_X[x_index + l] * (sigmoid_tmp[(tasklet_id*rows_per_cache) + row_index] - \
                        (cache_Y[y_index])<< SHIFT_AMOUNT) >> (OVERFLOW_SHIFT + SHIFT_AMOUNT); 
                    #endif
                }
            # if PRINT
            if(row_index < 100) {
                printf("dot_product dpu: %d, sigmoid_dpu: %d\n", dot_product_t, sigmoid); 
                printf("X at DPU: "); 
                for (uint32_t each_attribute = 0; each_attribute < n_size_pad; each_attribute++) {
                    printf("%d, ", cache_X[x_index+each_attribute]); 
                }
                printf("\n");
            }
            # endif
            }
            x_index += n_size; 
        } // end cache_X 
    } // accessed all rows 

    // Barrier
    barrier_wait(&my_barrier);

    // Reduction 
    // if (tasklet_id > 0) {
    //     for (unsigned int each_attribute = 0; each_attribute < n_size; each_attribute++) {
    //         gradient_tmp[each_attribute] += gradient_tmp[tasklet_id*n_size_pad + each_attribute]; 
    //     }
    // }
    // barrier_wait(&my_barrier);
    // if (tasklet_id == 0) {
    //     // partial result of gradient in this DPU
    //     mram_write((const void *) gradient_tmp, (__mram_ptr void *) DPU_RESULTS, n_size_pad_byte); 
    // }
    // printf("teask id %d",  tasklet_id);
    // printf("op_mode id %d",  op_mode);

    if (tasklet_id == 0) {
        
        if(op_mode == 0){
            int count=0;
            // T *arr;
            // T *globArr;
            // printf("check1 %d", NR_TASKLETS);
            for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS; each_tasklet++){
                // printf("hi \n");
                for (unsigned int row = 0; row < DPU_INPUT_ARGUMENTS.rows_per_tasklet[each_tasklet]; row++){
                    // printf("bye \n");
                // arr[count] = dot_product_temp[each_tasklet][row];
                dot_product_t[count]= dot_product_temp[each_tasklet][row];
                // sigmoid_tmp[count] = dot_product_t[count];
                // printf("%d  ",dot_product_temp[each_tasklet][row]);
                count++;
               }
            }
            // mram_write((const void *) dot_product_t, (__mram_ptr void *) DPU_PRODUCT,nr_rows );
        }
        else if(op_mode == 1){
            for (unsigned int each_tasklet = 1; each_tasklet < NR_TASKLETS; each_tasklet++){
                for (unsigned int each_attribute = 0; each_attribute < n_size; each_attribute++) {
                    gradient_tmp[each_attribute] += gradient_tmp[each_tasklet*n_size_pad + each_attribute]; 
                }
            }
            // partial result of gradient in this DPU
            mram_write((const void *) gradient_tmp, (__mram_ptr void *) DPU_RESULTS, n_size_pad_byte); 
        }
    }
    
    return 0;
}
