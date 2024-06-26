 
/**
 * app.c
 * GEMV Host Application Source File
 *
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
#if VERIF
	//m_size++;// increase the matrix by one for verification tags if it is part of the matrix
#endif
#if INTG
	//n_size++; 
#endif
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
	printf("group:%ld\n",group_number);
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
	printf("initialization done\n");

	// Timer
	Timer timer;
	Timer1 timer1;

	/*********************************verification************************************/
	// an emulation of the security scheme
	uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	struct AES_ctx ctx;
	float sec=0;
	uint8_t s1[]={ 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	
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
	#pragma omp parallel for
	  for (unsigned int i = 0; i < (n_size); i++){
	        firstTag1[lay][i] =0; 
    	}
    }

    startTimer(&timer1);

    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
	    s1[0]= (uint8_t)(A[lay]);
	    //#pragma omp parallel for
   	    for(unsigned int i = 0; i < (m_size*n_size); i++){ 
		T mul = pow(s1[0],((n_size)-(i%n_size))) * A[lay][i];
       	 	firstTag1[lay][i%(n_size)] += mul;        
   	    }
	    //#pragma omp parallel for
		for(unsigned int b=0; b< batch_size; b++){
            for (unsigned int i = 0; i < (n_size); i++){
                verifTag[lay][b] += (firstTag1[lay][i] * B[b][i]);
            }
		
		}
	}
     stopTimer(&timer1);
     sec = getElapsedTime(timer1);

#endif
	/*********************************verification************************************/
	
	
	/*********************************SecNDP ***************************************/

	uint8_t* first;
		first = malloc(n_size_pad * sizeof(uint8_t));

	T** ciphertext;
	ciphertext = malloc(batch_size * sizeof(T*));
	for(unsigned int b=0; b< batch_size; b++){
		ciphertext[b] = malloc(n_size_pad * sizeof(T));
	}
	T** ciphertexttemp;
	ciphertexttemp = malloc(batch_size * sizeof(T*));
	for(unsigned int b=0; b< batch_size; b++){
		ciphertexttemp[b] = malloc(n_size_pad * sizeof(T));
	}
	T1* temp = malloc(n_size_pad * sizeof(T));
	T * C_host;
	C_host = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	srand(0);
	for(unsigned int i=0; i< max_rows_per_dpu * nr_of_dpus; i++){
		C_host[i] = rand();
	}
	startTimer(&timer1);
	
	#pragma omp parallel for
	for(unsigned int i=0;i< n_size_pad; i++){
		first[i] = (uint8_t)(B+(i*sizeof(T)));
	}
	AES_init_ctx(&ctx, key);
	AES_ECB_encrypt(&ctx, first);
	for(unsigned int b=0; b< batch_size; b++){
		#pragma omp parallel for
		for(unsigned int i=0;i<n_size_pad; i++){
			temp[i] = (T1)first[i];
			ciphertext[b][i]= B[b][i] - temp[i];
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
	// startTimer(&timer1);

	for(unsigned int lay = 0; lay < NUM_LAYERS; lay++){
		for(unsigned int b=0 ; b< batch_size;b++){
			gemv(A, B[b], m_size, n_size,C[b], lay);
			for (unsigned int i = 0; i < m_size; i++)
			{
				if(C[b][i] <= 0) C[b][i] = 0;
			}
		}
	}
	// stopTimer(&timer1);
    // float cpuplain = getElapsedTime(timer1);

	stop(&timer, 0);

	float cpu[NUM_LAYERS]={0};
	float temp_gen[NUM_LAYERS]={0};
	float cpudpu[NUM_LAYERS]={0};
	float interdpu[NUM_LAYERS]={0};
	float kernel[NUM_LAYERS]={0};
	float dpucpu[NUM_LAYERS]={0};
	float merge=0;
	// T1* temp2 = malloc(n_size_pad  * sizeof(T1));
	T** C_total;
	C_total = malloc(batch_size * sizeof(T*));
	for(unsigned int b=0; b< batch_size; b++){
		C_total[b] = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	}
	uint64_t counter=0;
	uint64_t counter2 =0;
	uint8_t* first2 = malloc(n_size_pad * sizeof(uint8_t));
	//storing weights in UPMEM 
	//Since it is fixed for all the users and batches we send them to UPMEM offline
	for(int lay=0; lay <  NUM_LAYERS; lay++){
		
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, A[lay] + ((dpu_info[i].prev_rows_dpu * n_size))));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * (n_size_pad) * sizeof(T), DPU_XFER_DEFAULT));//lay * max_rows_per_dpu * (n_size_pad) * sizeof(T), max_rows_per_dpu * (n_size_pad) * sizeof(T), DPU_XFER_DEFAULT));
	}

	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		for (unsigned int g = 0; g<= (batch_size-1)/group_number; g++){
				i = 0;
				if (rep >= p.n_warmup)
					start(&timer, 1, rep - p.n_warmup);//TIMER, 1 is for cpu-dpu communication
					
				// Input arguments
				if (rep >= p.n_warmup)
						{
					startTimer(&timer1);
				}
				i = 0;
				DPU_FOREACH(dpu_set, dpu, i) {
					// Copy input arguments to DPU
					input_args[i].max_rows = max_rows_per_dpu;

					DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
				}
				DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

				//send current layer
				DPU_FOREACH(dpu_set, dpu, i) {
					int32_t lay1 = 0;		
					DPU_ASSERT(dpu_prepare_xfer(dpu, &lay1));
				}
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
				//end of CPU-DPU communication

				/********************************CPU computation in parallel to UPMEM ***********************************/
				T* dectemp ;
				dectemp = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
				if (rep >= p.n_warmup) startTimer(&timer1);
				for( int g =0; g< group_number; g++){
					// #pragma omp parallel for
					for(unsigned int i=0;i< (max_rows_per_dpu * nr_of_dpus); i++){
						dectemp[i] = (uint8_t)(B+(i*sizeof(T)));
					}
					AES_init_ctx(&ctx, key);
					AES_ECB_encrypt(&ctx, dectemp); 
					// gemv1(A[0],first2, m_size, n_size, C_host);
					for(unsigned int i=0; i< max_rows_per_dpu * nr_of_dpus; i++){
						C_host[i] = dectemp[i] + C_host[i];
					}
				}
				if (rep >= p.n_warmup){
					stopTimer(&timer1);
					cpu[0] += getElapsedTime(timer1);
				}
				
				/******************************** end of added cpu portion ************************/
				
				// Run kernel on DPUs
				if (rep >= p.n_warmup)
				{
			#if ENERGY
					DPU_ASSERT(dpu_probe_start(&probe));
			#endif
					startTimer(&timer1);
					start(&timer, 2, rep - p.n_warmup);
				}

				DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
				if (rep >= p.n_warmup){
					stop(&timer, 2);
					stopTimer(&timer1);
					kernel[0] += getElapsedTime(timer1);
					
			#if ENERGY
					DPU_ASSERT(dpu_probe_stop(&probe));
			#endif
				}

				//START LAYER 2 AND BEYOND
				//timer 4 is for inter DPU-CPU Communication
				for(int lay = 1; lay < NUM_LAYERS; lay++){
					if (rep >= p.n_warmup){
						start(&timer, 4, rep - p.n_warmup);
						startTimer(&timer1);		
					}

					i = 0;
					// Copy C_dpu //THIS MEAN GETTING THE OUTPUT OF LAYER 1 TO MAKE LAYER 2 INPUT
					DPU_FOREACH(dpu_set, dpu, i) {
						DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu[(i%group_number)] + i * max_rows_per_dpu));
					}
					DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, (NUM_LAYERS * ( max_rows_per_dpu * n_size_pad * sizeof(T))) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));

					if (rep >= p.n_warmup){
						stopTimer(&timer1);
						dpucpu[lay-1] += getElapsedTime(timer1);
					}
					/*************************combine results*************************/
					if (rep >= p.n_warmup)
					{
						startTimer(&timer1);
					}
					int partition=100;
					#pragma omp parallel for
					for(unsigned int i=0; i<max_rows_per_dpu * nr_of_dpus/partition ; i++){
						//#pragma omp parallel for
						for(unsigned int b=0; b< group_number;b++){
							//#pragma omp parallel for
							for(unsigned int part= i*100; part <= (i+1)*100; part++){
								C_total[b][part]= C_host[part] + C_dpu[b][part];
								if( C_total[b][part] <= 0)
									C_total[b][part]=0;
							}
						}
					}
					if (rep >= p.n_warmup){
						stopTimer(&timer1);
						interdpu[lay-1] += getElapsedTime(timer1);
					}

					/*************************End combine results*************************/
					
					// Sending the next layer input back to UPMEM
					i = 0;
					if (rep >= p.n_warmup)
					{
						startTimer(&timer1);
						start(&timer, 1, rep - p.n_warmup);
					}
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
						stop(&timer, 4);
					}
					if (rep >= p.n_warmup)
					{
			#if ENERGY
						DPU_ASSERT(dpu_probe_start(&probe));
			#endif
						startTimer(&timer1);
						start(&timer, 2, rep - p.n_warmup);
					}

					DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

					if (rep >= p.n_warmup)
					{
						stop(&timer, 2);
						stopTimer(&timer1);
						kernel[lay]+= getElapsedTime(timer1);
			#if ENERGY			
						DPU_ASSERT(dpu_probe_stop(&probe));
			#endif
					}
					/**********************cpu part***************************/
					
					/********************************add cpu portion ***********************************/
					if (rep >= p.n_warmup) startTimer(&timer1);
					for( int g =0; g< group_number; g++){
						// #pragma omp parallel for
						for(unsigned int i=0;i< (max_rows_per_dpu * nr_of_dpus); i++){
							dectemp[i] = (uint8_t)(B+(i*sizeof(T)));
						}
						AES_init_ctx(&ctx, key);
						AES_ECB_encrypt(&ctx, dectemp); 
						// gemv1(A[0],first2, m_size, n_size, C_host);
						for(unsigned int i=0; i< max_rows_per_dpu * nr_of_dpus; i++){
							C_host[i] = dectemp[i] + C_host[i];
						}
					}
					if (rep >= p.n_warmup){
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
				DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu[(i%group_number)] + i * max_rows_per_dpu));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
			if(rep >= p.n_warmup)
				stop(&timer, 3);
			
			/*************************combine results*************************/
			startTimer(&timer1);
			int partition=100;
			#pragma omp parallel for
			for(unsigned int i=0; i<max_rows_per_dpu * nr_of_dpus/partition ; i++){
				for(unsigned int b=0; b< group_number;b++){
					for(unsigned int part= i*100; part<= (i+1)*100; part++){
						C_total[b][part]= C_host[part] + C_dpu[b][part];
						if( C_total[b][part] <= 0)
							C_total[b][part]=0;
					}
				}
			}
			stopTimer(&timer1);
			interdpu[NUM_LAYERS-1] += getElapsedTime(timer1);
			merge += getElapsedTime(timer1);
			
			/*************************combine results*************************/

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
	
    /************************verify*********************************/
    float verif =0;
    // float verift =0;
    s1[0]=(uint8_t)(A[NUM_LAYERS-1]);
    //T1 verift=0;
    T **firstTag2 = malloc((NUM_LAYERS)*sizeof(T*)); 
    for(int lay=0 ;lay< NUM_LAYERS; lay++){
	firstTag2[lay] = malloc(batch_size* sizeof(T));
	}
	//float verif =0;
#if VERIF
    
	startTimer(&timer1);
	for(unsigned int b=0; b< batch_size; b++){
		firstTag2[NUM_LAYERS-1][b]=0;
		T pow = 1;
		// #pragma omp parallel for
		for (unsigned int i = 0; i < n_size; i++){
			pow *= s1[0];
			firstTag2[NUM_LAYERS-1][b] += (C_total[b][i]) * pow;
		}
		if (verifTag[NUM_LAYERS-1][b]==firstTag2[NUM_LAYERS-1][b]){
			int flag = 1;
		}
	}
	stopTimer(&timer1);
	verif = getElapsedTime(timer1);
#endif		
	/***************************edn verification*********************/
	// Print timing results

	printf("\nCPU Version Time (ms): ");
	print(&timer, 0, 1);

	printf("\n\nCPU-DPU Time (ms):");
	print(&timer, 1, p.n_reps);
	for(int i=0; i< NUM_LAYERS ; i++){
		printf("\n lay[%d] = %f:",i, (cpudpu[i]/p.n_reps)*1e3);
	}
	printf("\nDPU Kernel Time (ms):");
	print(&timer, 2, p.n_reps);
	for(int i=0; i< NUM_LAYERS ; i++){
                printf("\n lay[%d] = %f:",i, (kernel[i]/p.n_reps)*1e3);
        }

	printf("\nDPU-CPU Time (ms): ");
	print(&timer, 3, p.n_reps);
	for(int i=0; i< NUM_LAYERS ; i++){
                printf("\n lay[%d] = %f:",i, (dpucpu[i]/p.n_reps)*1e3);
        }

	printf("\nTotal Inter-DPU Time (ms): ");
	print(&timer, 4, p.n_reps);
	for(int i=0; i< NUM_LAYERS ; i++){
                printf("\n lay[%d] = %f:",i, (interdpu[i]/p.n_reps)*1e3);
    	}
	printf("\nSec Cal Time(ms): %f \nMerge Time(ms): %f\n  verification time: %f\n", sec*1e3, (merge/p.n_reps)*1e3, verif*1e3);
	for(int i=0; i< NUM_LAYERS ; i++){
                printf("\n CPU__lay[%d] = %f:",i, (cpu[i]/p.n_reps*1e3));
        }

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
	//free(A);
	//free(B);
	//free(C);
	//free(C_dpu);
	//free(C_host);
	//free(C_total);
	//free(temp);
	//free(ciphertext);
	//free(first);
#if VERIF 
	//free(firstTag1);
#endif
#if INTG
	//free(firstTag);
#endif
	DPU_ASSERT(dpu_free(dpu_set));

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

	return status ? 0 : -1;
}
