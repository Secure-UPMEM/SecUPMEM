 
/**
 * app.c
 * GEMV Host Application Source File
 *
 * This is only for performance purposes 
 * Please visit https://github.com/CMU-SAFARI/prim-benchmarks for the baseline implementation since our work is built upon it.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dpu.h>
#include <dpu_log.h>
#include <unistd.h>
#include <getopt.h>
#include <assert.h>
#include<math.h>
#include "gemv_utils.h"
#if ENERGY
#include <dpu_probe.h>
#endif

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"
#include "../support/aes.h"

// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/gemv_dpu"
#endif
#define NM_PART 1

static T** A;
static T** B;
T* B_temp;
static T** C;
static T** C_dpu;
uint64_t group_number;
// Create input arrays
static void init_data(T** A, T** B, unsigned int m_size, unsigned int n_size, unsigned int batch) {
	srand(0);
	for(int lay = 0; lay < NUM_LAYERS; lay++){
		#pragma omp parallel for
		for (unsigned int m = 0; m < m_size; m++) {
			for (unsigned int n = 0; n < n_size; n++)
			{
				A[lay][m * n_size + n] = (unsigned int) (rand()%50);
			}
		}
	}
	srand(0);
	for(unsigned int b = 0; b < batch ; b++){
		#pragma omp parallel for
		for (unsigned int i = 0; i < n_size; i++)
		{
			B[b][i] = (unsigned int) (rand()%25);
		}
	}
}



void gemv(T** A, T* B, unsigned int m_size, unsigned int n_size, T* C, int lay) {//may change
	#pragma omp parallel for
	for (unsigned int m = 0; m < m_size; m++) {
		C[m]=0;
		for (unsigned int n = 0; n < n_size; n++)
		{
			C[m] += A[lay][m * n_size + n] * B[n];
		}
	}
}

void gemv1(T1* A, uint8_t* B, unsigned int m_size, unsigned int n_size, T* C) {
	#pragma omp parallel for
	for (unsigned int m = 0; m < m_size; m++) {
		C[m] = 0;
		for (unsigned int n = 0; n < n_size; n++)
		{
			C[m] += A[m * n_size + n] * B[n];
		}
	}
}

// Main of the Host Application
int main(int argc, char **argv) {

	struct Params p = input_params(argc, argv);

	struct dpu_set_t dpu_set, dpu;
	uint32_t nr_of_dpus;

	// Allocate DPUs and load binary
	DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
	DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
	DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));

#if ENERGY
	struct dpu_probe_t probe;
	DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

	unsigned int i;
	unsigned int m_size = p.m_size;
	unsigned int n_size = p.n_size;
	unsigned int batch_size = p.batch;

	// Initialize help data
	dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t));
	dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
	uint32_t max_rows_per_dpu = 0;
	uint32_t n_size_pad = n_size;
	if(n_size % 2 == 1)
	{
		n_size_pad++;
	}
	uint64_t inputsize= n_size * nr_of_dpus*sizeof(T);
	uint64_t outputsize= m_size * nr_of_dpus*sizeof(T);
	uint64_t mat= n_size*m_size;
	uint64_t numsize= NUM_LAYERS * sizeof(T);
	uint64_t totmat = mat * numsize;
	uint64_t matrixsize= totmat;
	uint64_t mram_sie=65000000;
	uint64_t bala = mram_sie * NR_DPUS;
	group_number= (uint64_t)((bala) / (inputsize+outputsize+matrixsize));
	if (group_number >= batch_size) group_number = batch_size; 
	i = 0;
	// printf("group:%ld\n",group_number);
	uint32_t nr_dpus_group = nr_of_dpus/group_number;
	DPU_FOREACH(dpu_set, dpu, i) {
		uint32_t rows_per_dpu;
		uint32_t prev_rows_dpu = 0;

		uint32_t chunks = (uint32_t)((float)m_size / (float)nr_dpus_group);
		rows_per_dpu = chunks;
		uint32_t rest_rows = m_size % nr_dpus_group;
		if (i < rest_rows)
			rows_per_dpu++;
		if (rest_rows > 0) {
			if (i >= rest_rows)
				prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
			else
				prev_rows_dpu = i * (chunks + 1);
		} else {
			prev_rows_dpu = i * chunks;
		}
		// Keep max rows for parallel transfers
		uint32_t rows_per_dpu_pad = rows_per_dpu;
		if (rows_per_dpu_pad % 2 == 1) // 4-byte elements
			rows_per_dpu_pad++;
		if (rows_per_dpu_pad > max_rows_per_dpu)
			max_rows_per_dpu = rows_per_dpu_pad;

		dpu_info[i].rows_per_dpu = max_rows_per_dpu;//rows_per_dpu;
		dpu_info[i].rows_per_dpu_pad = max_rows_per_dpu;//rows_per_dpu_pad; //may change
		dpu_info[i].prev_rows_dpu = prev_rows_dpu;
		
		input_args[i].n_size = n_size;
		input_args[i].n_size_pad = n_size_pad;
		input_args[i].nr_rows = max_rows_per_dpu;
	}
	
	printf("Initialization\n");
	A = malloc(NUM_LAYERS * sizeof(T*));
	for(int lay=0; lay< NUM_LAYERS; lay++){
		A[lay] = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
	}
	B = malloc(batch_size * sizeof(T*));
	for(unsigned int b=0; b< batch_size; b++){
		B[b] = malloc(n_size_pad * sizeof(T));
	}
	C = malloc(batch_size * sizeof(T*));
	for(unsigned int b=0; b< batch_size; b++){
		C[b] = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	}

	// Initialize data with arbitrary data
	init_data(A, B, m_size, n_size, batch_size);

	// Timer
	Timer timer;
	Timer1 timer1;

	/*********************************verification************************************/
	// an emulation of the security scheme you can also use AES CTR - We create a counter and then encrypt it to generate random numbers
	uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	struct AES_ctx ctx;
	float sec=0;
	
#if VERIF
    T **firstTag1 = malloc((NUM_LAYERS)*sizeof(T*)); 
    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
   	firstTag1[lay] = malloc((n_size)*sizeof(T)); 
    }
    T **verifTag = malloc((NUM_LAYERS)*sizeof(T*)); 
    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
         verifTag[lay] = malloc((batch_size)*sizeof(T)); 
    }

    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
	// #pragma omp parallel for
	  for (unsigned int i = 0; i < (n_size); i++){
	        firstTag1[lay][i] =0; 
    	}
    }

    // startTimer(&timer1);

    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
	    
		T seed = 2;//(uint8_t)(A[lay]);
		T pow = 1;
		//#pragma omp parallel for
   	    for(unsigned int i = 0; i < (m_size); i++){ 
			// firstTag1[lay][inside] =0;
			for(unsigned int inside = 0; inside < (n_size); inside++){
				firstTag1[lay][inside] += ( A[lay][i * m_size + inside]) * pow;
			}   
			if(pow == 64) pow = 1;
			else pow *= seed; 
   	    }
	}
	for(unsigned int b=0; b< batch_size; b++){
		verifTag[0][b] =  0 ;
		//#pragma omp parallel for
        for (unsigned int i = 0; i < (n_size); i++){
            verifTag[0][b] += (firstTag1[0][i] * B[b][i]);
        }
	}
    //  stopTimer(&timer1);
    //  sec = getElapsedTime(timer1);

#endif
	/*********************************verification************************************/
	
	
	/*********************************SecNDP ***************************************/

	uint8_t* first = malloc(n_size_pad * sizeof(uint8_t));

	T** ciphertext;
	ciphertext = malloc(batch_size * sizeof(T*));
	T** ciphertexttemp;
	ciphertexttemp = malloc(batch_size * sizeof(T*));

	for(unsigned int b=0; b< batch_size; b++){
		ciphertext[b] = malloc(n_size_pad * sizeof(T));
		ciphertexttemp[b] = malloc(n_size_pad * sizeof(T));
	}
	
	T * C_host;
	C_host = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));

	startTimer(&timer1);
	
	// #pragma omp parallel for
	for(unsigned int i=0;i< n_size_pad; i++){
		first[i] = (uint8_t)(20+(i*sizeof(T)));
	}
	AES_init_ctx(&ctx, key);
	AES_ECB_encrypt(&ctx, first);
	for(unsigned int b=0; b< batch_size; b++){
		// #pragma omp parallel for
		for(unsigned int i=0;i<n_size_pad; i++){
			ciphertext[b][i]= B[b][i] - (T1)first[i];
			ciphertexttemp[b][i]=ciphertext[b][i];
		}
	}
    stopTimer(&timer1);
    sec += getElapsedTime(timer1);
	/*********************************SecNDP****************************************/
	
	C_dpu = malloc(batch_size * sizeof(T*));
	for(unsigned int b=0; b< batch_size; b++){
		C_dpu[b] = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	}
	// Compute output on CPU (performance comparison and verification purposes)
	start(&timer, 0, 0);

	for(unsigned int lay = 0; lay < NUM_LAYERS; lay++){
		for(unsigned int b=0 ; b< batch_size;b++){
			gemv(A, B[b], m_size, n_size,C[b], lay);
			for (unsigned int i = 0; i < m_size; i++)
			{
				if(C[b][i] <= 0) C[b][i] = 0;
			}
		}
	}

	stop(&timer, 0);

	float cpu[NUM_LAYERS]={0};
	float cpudpu[NUM_LAYERS]={0};
	float interdpu[NUM_LAYERS]={0};
	float kernel[NUM_LAYERS]={0};
	float dpucpu[NUM_LAYERS]={0};
	float merge=0;
	
	T** C_total;
	C_total = malloc(batch_size * sizeof(T*));
	for(unsigned int b=0; b< batch_size; b++){
		C_total[b] = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	}
	uint8_t* first2 = malloc(n_size_pad * sizeof(uint8_t));
	
	//storing weights in UPMEM 
	//Since it is fixed for all the users and batches we send them to UPMEM offline
	for(int lay=0; lay <  NUM_LAYERS; lay++){
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, A[lay] + ((dpu_info[(i/group_number)].prev_rows_dpu * n_size))));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, lay * max_rows_per_dpu * (n_size_pad) * sizeof(T), max_rows_per_dpu * (n_size_pad) * sizeof(T), DPU_XFER_DEFAULT));//, max_rows_per_dpu * (n_size_pad) * sizeof(T), DPU_XFER_DEFAULT));
	}

	T **firstTag2 = malloc((NUM_LAYERS)*sizeof(T*)); 
    for(int lay=0 ;lay< NUM_LAYERS; lay++){
		firstTag2[lay] = malloc(batch_size* sizeof(T));
	}
	float verif =0;

	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		for (unsigned int round = 0; round<= ((batch_size-1)/group_number); round++){
			i = 0;
			if (rep >= p.n_warmup){
				start(&timer, 1, rep - p.n_warmup);//TIMER, 1 is for cpu-dpu communication
				startTimer(&timer1);
			}
			i = 0;
			DPU_FOREACH(dpu_set, dpu, i) {
				// Copy input arguments to DPU
				input_args[i].max_rows = max_rows_per_dpu;

				DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
			i = 0;
			//send current layer
			DPU_FOREACH(dpu_set, dpu, i) {
				int32_t lay1 = 0;		
				DPU_ASSERT(dpu_prepare_xfer(dpu, &lay1));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "CURRENT_LAYER", 0, sizeof(uint32_t), DPU_XFER_DEFAULT));
			i = 0;
			//sending ciphertext to cpu
			//we need to send a group of batch to the UPMEM for parallel computation
			DPU_FOREACH(dpu_set, dpu, i) {
				DPU_ASSERT(dpu_prepare_xfer(dpu, ciphertext[(i%group_number)]));// ciphertext here is part of the B which is the vector
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, NUM_LAYERS * (max_rows_per_dpu * (n_size_pad) * sizeof(T)) , n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
			
			if (rep >= p.n_warmup){
				stop(&timer, 1);
				stopTimer(&timer1);
				cpudpu[0] += getElapsedTime(timer1);
			}
			
			/********************************CPU computation in parallel to UPMEM ***********************************/

			if (rep >= p.n_warmup) {
				
				startTimer(&timer1);
				start(&timer, 9, rep - p.n_warmup);
			}
			for(unsigned int g = 0; g < group_number; g++){
				// #pragma omp parallel for
				for(unsigned int i=0;i< (n_size_pad); i++){
					first2[i] = (uint8_t)(20+(i*sizeof(T)));
				}
				AES_init_ctx(&ctx, key);
				AES_ECB_encrypt(&ctx, first2); 
				gemv1(A[0],first2, m_size, n_size, C_host);
			}
			if (rep >= p.n_warmup){
				stop(&timer, 9);
				stopTimer(&timer1);
				cpu[0] += getElapsedTime(timer1);
			}
			
			/******************************** end of added cpu portion ************************/
				
			// Run kernel on DPUs
			if (rep >= p.n_warmup)
			{
				start(&timer, 2, rep - p.n_warmup);
				startTimer(&timer1);
			}

			i=0;
			DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
			if (rep >= p.n_warmup){
				stopTimer(&timer1);
				kernel[0] += getElapsedTime(timer1);
				stop(&timer, 2);
			#if ENERGY
					DPU_ASSERT(dpu_probe_stop(&probe));
			#endif
				}

			//START LAYER 2 AND BEYOND
			//timer 4 is for inter DPU-CPU Communication
			for(int lay = 1; lay < NUM_LAYERS; lay++){
				if (rep >= p.n_warmup){
					start(&timer, 3, rep - p.n_warmup);
					start(&timer, 4, rep - p.n_warmup);
					startTimer(&timer1);		
				}

				i = 0;
				// Copy C_dpu //THIS MEAN GETTING THE OUTPUT OF LAYER 1 TO MAKE LAYER 2 INPUT
				DPU_FOREACH(dpu_set, dpu, i) {
					DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu[(i%group_number)] + (i/group_number) * max_rows_per_dpu));
				}
				DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, (NUM_LAYERS * ( max_rows_per_dpu * n_size_pad * sizeof(T))) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));

				if (rep >= p.n_warmup){
					stop(&timer, 3);
					stopTimer(&timer1);
					dpucpu[lay-1] += getElapsedTime(timer1);
				}
				/*************************combine results*************************/
				if (rep >= p.n_warmup)
				{
					startTimer(&timer1);
					start(&timer, 8, rep - p.n_warmup);
				}
				// int partition=100;
				// #pragma omp parallel for
				// for(unsigned int i=0; i<max_rows_per_dpu * nr_of_dpus/partition ; i++){
				// 	//#pragma omp parallel for
				// 	for(unsigned int b=0; b< group_number;b++){
				// 		//#pragma omp parallel for
				// 		for(unsigned int part= i*100; part <= (i+1)*100; part++){
				// 			C_total[b][part]= C_host[part] + C_dpu[b][part];
				// 			if( C_total[b][part] <= 0)
				// 				C_total[b][part]=0;
				// 		}
				// 	}
				// }
				for(unsigned int b=0; b< group_number;b++){
					for(unsigned int i=0; i<m_size ; i++){
						C_total[b][i]= C_host[i] + C_dpu[b][i];
						if( C_total[b][i] < 0)
								C_total[b][i]=0;
					}
				}

				for(unsigned int b=0; b< group_number; b++){
					verifTag[lay][b] =  0 ;
					//#pragma omp parallel for
					for (unsigned int i = 0; i < (n_size); i++){
						verifTag[lay][b] += (firstTag1[lay][i] * C_total[b][i]);

					}
				}

				if (rep >= p.n_warmup){
					stopTimer(&timer1);
					interdpu[lay-1] += getElapsedTime(timer1);
				}

				/*************************End combine results*************************/
				
				/************************verify*********************************/
				#if VERIF
				int flag = 1;
				startTimer(&timer1);
				T seed=2;//(uint8_t)(A[lay-1]);
				for(unsigned int b=0; b< group_number; b++){
					firstTag2[lay-1][b]=0;
					T pow = 1;
					// #pragma omp parallel for
					for (unsigned int i = 0; i < m_size; i++){
						firstTag2[lay-1][b] += (C_total[b][i]) * pow;
						if(pow == 64) pow = 1;
						else pow *= seed; 
					}
					// if (verifTag[lay-1][b]!=firstTag2[lay-1][b]){ // can be performed in parallel with next layer computation
					// 	flag = 0;
					// }
				}
				// if(flag ==1) printf("verified layer: %d\n", lay-1); //for now its commented since we use randomly generated numbers for OTPs.
				stopTimer(&timer1);
				verif += getElapsedTime(timer1);
				#endif	
				uint8_t* first1;
				first1 = malloc(n_size_pad * sizeof(uint8_t));
				//encryption
				// #pragma omp parallel for
				for(unsigned int i=0;i< n_size_pad; i++){
					first1[i] = (uint8_t)(20+(i*sizeof(T))); 
				}

				AES_init_ctx(&ctx, key);
				AES_ECB_encrypt(&ctx, first1);
				for(unsigned int b=0; b< group_number; b++){
					// #pragma omp parallel for
					for(unsigned int i=0;i<n_size_pad; i++){
						C_total[b][i] = C_total[b][i] - (T1)first1[i];
					}
				}
				free(first1);

					/***************************edn verification*********************/


				// Sending the next layer input back to UPMEM
				i = 0;
				if (rep >= p.n_warmup)
				{
					stop(&timer, 8);
					stop(&timer, 4);
					startTimer(&timer1);
					start(&timer, 1, rep - p.n_warmup);
				}
				i = 0;
				//send current layer
				DPU_FOREACH(dpu_set, dpu, i) {
					int32_t lay1 = lay;		
					DPU_ASSERT(dpu_prepare_xfer(dpu, &lay1));
				}
				DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "CURRENT_LAYER", 0, sizeof(int32_t), DPU_XFER_DEFAULT));
				// I need to add privacy here.
				i = 0;
				DPU_FOREACH(dpu_set, dpu, i) {
					DPU_ASSERT(dpu_prepare_xfer(dpu, C_total[(i%group_number)]));
				}
				DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, NUM_LAYERS * (max_rows_per_dpu * (n_size_pad) * sizeof(T)) , n_size_pad * sizeof(T), DPU_XFER_DEFAULT));

				if (rep >= p.n_warmup){
					stopTimer(&timer1);
					cpudpu[lay] += getElapsedTime(timer1);
					stop(&timer, 1);
					// stop(&timer, 4);
				}
				if (rep >= p.n_warmup)
				{
					start(&timer, 2, rep - p.n_warmup);
			#if ENERGY
						DPU_ASSERT(dpu_probe_start(&probe));
			#endif
					startTimer(&timer1);
				}

				DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

				if (rep >= p.n_warmup)
				{
					stopTimer(&timer1);
					kernel[lay]+= getElapsedTime(timer1);
					stop(&timer, 2);
			#if ENERGY			
						DPU_ASSERT(dpu_probe_stop(&probe));
			#endif
				}
				/**********************cpu part***************************/
				
				/********************************add cpu portion ***********************************/
				if (rep >= p.n_warmup) {
					startTimer(&timer1);
					start(&timer, 9, rep - p.n_warmup);
				}
				for(unsigned int g = 0; g< group_number; g++){
					// #pragma omp parallel for
					for(unsigned int i=0;i< (n_size_pad); i++){
						first2[i] = (uint8_t)(10+(i*sizeof(T)));
					}
					AES_init_ctx(&ctx, key);
					AES_ECB_encrypt(&ctx, first2); 
					gemv1(A[lay],first2, m_size, n_size, C_host);
				}
				if (rep >= p.n_warmup){
					stop(&timer, 9);
					stopTimer(&timer1);
					cpu[lay] += getElapsedTime(timer1);
				}
			
			/******************************** end of added cpu portion ************************/
				/**********************cpu part***************************/
			}	
			//getting final result	
			//timer 3 dpu-cpu communication		
			if (rep >= p.n_warmup)
				start(&timer, 3, rep - p.n_warmup);
			i = 0;
			DPU_FOREACH(dpu_set, dpu, i) {
				DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu[(i%group_number)] + (i/group_number) * max_rows_per_dpu));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
			if(rep >= p.n_warmup)
				stop(&timer, 3);
			
			/*************************combine results*************************/
			startTimer(&timer1);
			if (rep >= p.n_warmup)
				start(&timer, 8, rep - p.n_warmup);
			
			// int partition=100;
			// #pragma omp parallel for
			// for(unsigned int i=0; i<max_rows_per_dpu * nr_of_dpus/partition ; i++){
			// 	for(unsigned int b=0; b< group_number;b++){
			// 		for(unsigned int part= i*100; part<= (i+1)*100; part++){
			// 			C_total[b][part]= C_host[part] + C_dpu[b][part];
			// 			if( C_total[b][part] <= 0)
			// 				C_total[b][part]=0;
			// 		}
			// 	}
			// }
			for(unsigned int b=0; b< group_number;b++){
				for(unsigned int i=0; i<m_size ; i++){
					C_total[b][i]= C_host[i] + C_dpu[b][i];
					if( C_total[b][i] < 0)
							C_total[b][i]=0;
				}
			}
			stopTimer(&timer1);
			interdpu[NUM_LAYERS-1] += getElapsedTime(timer1);
			merge += getElapsedTime(timer1);
			
			/*************************combine results*************************/
			/************************verify*********************************/
			#if VERIF
				int flag = 1;
				startTimer(&timer1);
				T seed= 2;//(uint8_t)(A[NUM_LAYERS-1]);
				for(unsigned int b=0; b< group_number; b++){
					firstTag2[NUM_LAYERS-1][b]=0;
					T pow = 1;
					// #pragma omp parallel for
					for (unsigned int i = 0; i < m_size; i++){
						firstTag2[NUM_LAYERS-1][b] += (C_total[b][i]) * pow;
						if(pow == 64) pow = 1;
						else pow *= seed; 
					}
					// if (verifTag[NUM_LAYERS-1][b]!=firstTag2[NUM_LAYERS-1][b]){// can be performed in parallel
					// 	flag = 0;
					// }
				}
				// if(flag ==1) printf("verified last layer\n"); // Since we used random values for tags, we cannot verify 
				stopTimer(&timer1);
				verif += getElapsedTime(timer1);
			#endif	
			if (rep >= p.n_warmup)
				stop(&timer, 8);	
			/***************************end verification*********************/
			

			// Display DPU Logs
			DPU_FOREACH(dpu_set, dpu) {
				DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
			}
			for(unsigned int b=0; b< group_number; b++){
				for(unsigned int i=0;i<  n_size_pad; i++){        	
					ciphertext[b][i]=ciphertexttemp[b][i];
				}
			}
		}
	}

#if ENERGY
	double acc_energy, avg_energy, acc_time, avg_time;
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_ACCUMULATE, &acc_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &avg_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_ACCUMULATE, &acc_time));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_AVERAGE, &avg_time));
#endif
	// Print timing results

	printf("\nCPU Version Time (ms): ");
	print(&timer, 0, 1);

	// printf("\n\nCPU-DPU Time (ms):");
	// print(&timer, 1, p.n_reps);
	// for(int i=0; i< NUM_LAYERS ; i++){
	// 	printf("\n lay[%d] = %f:",i, (cpudpu[i]/p.n_reps)*1e3);
	// }
	// printf("\nDPU Kernel Time (ms):");
	// print(&timer, 2, p.n_reps);
	// for(int i=0; i< NUM_LAYERS ; i++){
    //             printf("\n lay[%d] = %f:",i, (kernel[i]/p.n_reps)*1e3);
    //     }

	// printf("\nDPU-CPU Time (ms): ");
	// print(&timer, 3, p.n_reps);
	// for(int i=0; i< NUM_LAYERS ; i++){
    //             printf("\n lay[%d] = %f:",i, (dpucpu[i]/p.n_reps)*1e3);
    //     }

	// printf("\nTotal Inter-DPU Time (ms): ");
	// print(&timer, 4, p.n_reps);
	// for(int i=0; i< NUM_LAYERS ; i++){
    //             printf("\n lay[%d] = %f:",i, (interdpu[i]/p.n_reps)*1e3);
    // 	}
	// printf("\nSec Cal Time(ms): %f \nMerge Time(ms): %f\n  verification time: %f\n", sec*1e3, (merge/p.n_reps)*1e3, verif*1e3);
	// for(int i=0; i< NUM_LAYERS ; i++){
    //             printf("\n CPU__lay[%d] = %f:",i, (cpu[i]/p.n_reps*1e3));
    //     }

	float cpuside = (timer.time[9]) / (1000*p.n_reps);
    float dpuside = (timer.time[1]+timer.time[2]+timer.time[3]) / (1000*p.n_reps);
    float execution_time = fmax(cpuside,dpuside) + (timer.time[8])/ (1000*p.n_reps);
    printf("\nExecution time: %f ms\n\n", execution_time );

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif

	// Check output
	bool status = true;
	/*unsigned int n,j;
	i = 0;
	for (n = 0; n < nr_of_dpus; n++) {
		for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
			//printf(" %u, %u \n",C_total[n * max_rows_per_dpu + j],C[i]);
			if((n*dpu_info[n].rows_per_dpu)+j< m_size & C[i] != C_total[n * max_rows_per_dpu + j]) {
				status = false;
				//printf(" %u, %u \n",C_total[n * max_rows_per_dpu + j],C[i]);
#if PRINT
	//			printf("%d: %d -- %d\n", i, C[i], C_dpu[n * max_rows_per_dpu + j]);
#endif
			}
			//printf(" %u, %u \n",C_total[n * max_rows_per_dpu + j],C[i]);
			i++;
		}
	}
	if (status) {
		printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs and Tag are equal\n");
	} else {
		printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
	}*/

	// Deallocation
	for(int lay=0; lay< NUM_LAYERS; lay++){
		free(A[lay]);
		free(firstTag1[lay]);
		free(firstTag2[lay]);
		free(verifTag[lay]);
	}
	for(unsigned int b=0; b< batch_size; b++){
		free(B[b]);
		free(C[b]);
		// free(C_host[b]);
		free(C_dpu[b]);
		free(ciphertext[b]);
		free(ciphertexttemp[b]);
		free(C_total[b]);
	}
	free(B);
	free(C);
	free(C_dpu);
	free(C_host);
	free(firstTag1);
	free(firstTag2);
	free(verifTag);
	free(C_total);
	free(ciphertexttemp);
	free(ciphertext);
	free(first);
	free(first2);
	// free(dectemp);
	DPU_ASSERT(dpu_free(dpu_set));

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

	return status ? 0 : -1;
}
