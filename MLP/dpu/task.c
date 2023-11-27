/*
 * Matrix vector multiplication with multiple tasklet
 *
 */
#include <stdint.h>
#include <stdio.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <seqread.h>
#include <stdlib.h>
#include "../support/common.h"

#define roundup(n, m) ((n / m) * m + m)

__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host int32_t CURRENT_LAYER;
uint32_t s = 0x2b;
// GEMV
static uint32_t gemv(T *bufferC, T *bufferA, T *bufferB, int pos) {
	uint32_t result=0;
	for (unsigned int i = 0; i < BLOCK_SIZE / sizeof(T); i++) {	
		bufferC[pos] += bufferA[i] * bufferB[i];
		#if INTG
		result += (bufferA[i] * s);
		#endif
			
	}
	//printf("result:%d\n", result);
	return result;
}


// Barrier
BARRIER_INIT(my_barrier, NR_TASKLETS);

// main
int main() {
	unsigned int tasklet_id = me();
#if PRINT
	// printf("tasklet_id = %u\n", tasklet_id);
	//printf("lay = %d\n", CURRENT_LAYER);
#endif
	
	if (tasklet_id == 0){ // Initialize once the cycle counter
		mem_reset(); // Reset the heap
	//printf("T:%d", sizeof(T));
	}
	// Barrier
	
	barrier_wait(&my_barrier);

	int32_t n_size = DPU_INPUT_ARGUMENTS.n_size;
	int32_t n_size_pad = DPU_INPUT_ARGUMENTS.n_size_pad;
	uint32_t nr_rows = DPU_INPUT_ARGUMENTS.nr_rows;
	uint32_t max_rows = DPU_INPUT_ARGUMENTS.max_rows;

	unsigned int element_per_cacheC = 8/sizeof(T);

	unsigned int nrows = nr_rows;
	unsigned int rows_per_tasklet; 
	unsigned int start_row;
	unsigned int chunks = nrows / (NR_TASKLETS * element_per_cacheC);
	unsigned int dbl_chunks = chunks * element_per_cacheC; //chunks + chunks; 
	rows_per_tasklet = dbl_chunks;
	unsigned int rest_rows = nrows % (NR_TASKLETS * element_per_cacheC); //(NR_TASKLETS + NR_TASKLETS);
	
	if ((tasklet_id * element_per_cacheC) < rest_rows)
		rows_per_tasklet += element_per_cacheC;
	if (rest_rows > 0) {
		if ((tasklet_id * element_per_cacheC) >= rest_rows) {
			// unsigned int hlf_rest_rows = rest_rows >> 1;
			if ((rest_rows % element_per_cacheC) != 0)
				start_row = roundup(rest_rows, element_per_cacheC) + tasklet_id * dbl_chunks; 
				// start_row = (hlf_rest_rows + 1) * (dbl_chunks + 2) + (tasklet_id - 1 - hlf_rest_rows) * dbl_chunks;
			else
				start_row = rest_rows + tasklet_id * dbl_chunks; 
				// start_row = (hlf_rest_rows) * (dbl_chunks + 2) + (tasklet_id - hlf_rest_rows) * dbl_chunks;
		} else 
			start_row = tasklet_id * (dbl_chunks + element_per_cacheC);
			// start_row = tasklet_id * (dbl_chunks + 2);
	} else {
		start_row = tasklet_id * (dbl_chunks);
	}

	// Address of the current row in MRAM
	uint32_t* mram_base_addr_A[NUM_LAYERS];
	for( int i=0; i<NUM_LAYERS; i++){
		mram_base_addr_A[i] = (uint32_t) (DPU_MRAM_HEAP_POINTER + start_row * n_size * sizeof(T) + (i*max_rows * n_size_pad * sizeof(T)));
	}
	uint32_t mram_base_addr_B = (uint32_t) (DPU_MRAM_HEAP_POINTER + (NUM_LAYERS*max_rows * n_size_pad * sizeof(T)));
	uint32_t mram_base_addr_C = (uint32_t) (DPU_MRAM_HEAP_POINTER + (NUM_LAYERS*max_rows * n_size_pad * sizeof(T)) + n_size_pad * sizeof(T) + start_row * sizeof(T));
	uint32_t mram_temp_addr_A = mram_base_addr_A[0];
	uint32_t mram_temp_addr_B = mram_base_addr_B;

	// Inititalize a local cache to store the MRAM block
	T *cache_A = (T *) mem_alloc(BLOCK_SIZE + 8);
	T *cache_A_aux = (T *) mem_alloc(8);
	T *cache_B = (T *) mem_alloc(BLOCK_SIZE);
	T *cache_C = (T *) mem_alloc(8);

	int offset = 0;

	#if PRINT
	//printf("id: %d, rows_per_tasklet = %d\n",tasklet_id, rows_per_tasklet);
	//printf("id: %d, start_row = %d\n",tasklet_id, start_row);
	#endif

	// Iterate over nr_rows
	// for (unsigned int i = start_row; i < start_row + rows_per_tasklet; i += 2) {
	int k=0;
	uint32_t result=0;
	for (unsigned int i = start_row; i < start_row + rows_per_tasklet; i += element_per_cacheC) {

		mram_temp_addr_A = (uint32_t) (DPU_MRAM_HEAP_POINTER + i * n_size * sizeof(T) + (CURRENT_LAYER*max_rows * n_size_pad * sizeof(T)) );
		mram_temp_addr_B = mram_base_addr_B;

		// cache_C[0] = 0;
		// cache_C[1] = 0;

		// clear the cache
		for(unsigned int c = 0; c < element_per_cacheC; c++){
			cache_C[c] = 0; 
		}

		// for(unsigned int pos = 0; pos < 2 && i + pos < nr_rows; pos++){
		// for(unsigned int pos = 0; (pos < element_per_cacheC) && ((i + pos) < (start_row + rows_per_tasklet)); pos++){
		// for(unsigned int pos = 0; pos < element_per_cacheC && i + pos < nr_rows; pos++){ 
		for(unsigned int pos = 0; pos < element_per_cacheC; pos++){ 
			if(i + pos >= nr_rows){
				// printf("id: %d, nrows: %d, error\n", tasklet_id, nrows);
				break;
			} 

			int n = 0, j;
			for (n = 0; n < (int32_t) (n_size - (BLOCK_SIZE/sizeof(T))); n += (BLOCK_SIZE / sizeof(T)))
			{

				mram_read((__mram_ptr void const*) (mram_temp_addr_A), cache_A, BLOCK_SIZE);
				mram_read((__mram_ptr void const*) (mram_temp_addr_B), cache_B, BLOCK_SIZE);

				if(offset)
				{

					for(unsigned int off = 0; off < (BLOCK_SIZE / sizeof(T)) - 1; off++)
					{
						cache_A[off] = cache_A[off + 1];
					}

					mram_read((__mram_ptr void const*) (mram_temp_addr_A + BLOCK_SIZE), cache_A_aux, 8);

					cache_A[BLOCK_SIZE / sizeof(T) - 1] = cache_A_aux[0];
				}

				// Compute GEMV
				result += gemv(cache_C, cache_A, cache_B, pos);
				//printf("result:%d\n", result);
				//printf("gemv");
				
				// Update memory addresses
				mram_temp_addr_A += BLOCK_SIZE;
				mram_temp_addr_B += BLOCK_SIZE;
			}

			mram_read((__mram_ptr void const*) (mram_temp_addr_A), cache_A, BLOCK_SIZE);


			if(offset)
			{
				for(unsigned int off = 0; off < (BLOCK_SIZE / sizeof(T)) -1; off++)
				{

					cache_A[off] = cache_A[off + 1];
				}

				mram_read((__mram_ptr void const*) (mram_temp_addr_A + BLOCK_SIZE ), cache_A_aux, 8);

  			       cache_A[BLOCK_SIZE / sizeof(T) - 1] = cache_A_aux[0];
			}


			mram_read((__mram_ptr void const*) (mram_temp_addr_B), cache_B, BLOCK_SIZE);
			//uint32_t result=0;
			//printf("n %u, n: %u \n", n_size, n);
			for (j = 0; j < (int) (n_size - n); j++) {
				// Compute GEMV.
				
				//printf("result:%u , a: %u\n",result,cache_A[j]);
				//printf("result:%u , a: %u\n",result,cache_A[j]);
				if(j >= (int)(BLOCK_SIZE / sizeof(T))){ 
					printf("error\n");
					break;
				}
				cache_C[pos] += cache_A[j] * cache_B[j];
				//printf("cache a :
				//printf("tasklet:%d, pos:%d, result:%u , a: %u, B:%u\n",me(),pos,result,cache_A[j],cache_B[j]);
				#if INTG
				if(j == n_size-n-1) {
					//k++;
					if(cache_A[j] != result){
						//printf("Data Not Verified\n");
						//printf("tasklet:%d, pos:%d, result:%u , a: %u\n",me(),pos,result,cache_A[j]);		
					}
				result=0;
				}
				else 
					result += (cache_A[j] * s);
				#endif
			}

			
			mram_temp_addr_A += (BLOCK_SIZE - ((BLOCK_SIZE / sizeof(T)) - (n_size - n)) * sizeof(T));
			mram_temp_addr_B = mram_base_addr_B;

			if(mram_temp_addr_A % 8 != 0)
			{
				offset = 1;
			}
			else
			{
				offset = 0;
			}
		}
		// Write cache to current MRAM block
		
		mram_write(cache_C, (__mram_ptr void *) (mram_base_addr_C), 8);

		// Update memory address
		// mram_base_addr_C += 2 * sizeof(T);
		mram_base_addr_C += 8; 

	}

	return 0;
}
