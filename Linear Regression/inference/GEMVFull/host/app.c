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

static T* A;
static T* B;
static T* C;
static T* C_dpu;

// Create input arrays
static void init_data(T* A, T* B, unsigned int m_size, unsigned int n_size) {
	srand(0);
	for (unsigned int i = 0; i < m_size * n_size; i++)
	{
		A[i] = 0;
		
	}
	
	for (unsigned int m = 0; m < m_size-1; m++) {
		for (unsigned int n = 0; n < n_size; n++)
		{
			A[m * n_size + n] = (unsigned int) (rand()%50);
		}
	}
	
	/*for (unsigned int i = 0; i < (m_size * n_size)-n_size; i++)
	{
		A[i] = (unsigned int) (rand()%50);
		
	}*/
	srand(0);
	for (unsigned int i = 0; i < n_size; i++)
	{
		B[i] = (unsigned int) (rand()%25);
	}
	B[n_size-1] = 0; 
}

// Compute output in the host
static void gemv_host(T* C, T* A, T* B, unsigned int m_size, unsigned int n_size) {
	for (unsigned int i = 0; i < m_size; i++)
	{
		C[i] = 0;
	}

	for (unsigned int m = 0; m < m_size; m++) {
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
#if VERIF
	m_size++;
#endif
#if INTG
	n_size++; 
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

	i = 0;
	DPU_FOREACH(dpu_set, dpu, i) {
		uint32_t rows_per_dpu;
		uint32_t prev_rows_dpu = 0;
		uint32_t chunks = m_size / nr_of_dpus;
		rows_per_dpu = chunks;
		uint32_t rest_rows = m_size % nr_of_dpus;
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
		dpu_info[i].rows_per_dpu_pad = max_rows_per_dpu;//rows_per_dpu_pad;
		dpu_info[i].prev_rows_dpu = prev_rows_dpu;
		//printf("max:%d , prev:%d\n",max_rows_per_dpu,prev_rows_dpu);
		// Copy input arguments to DPU
		input_args[i].n_size = n_size;
		input_args[i].n_size_pad = n_size_pad;
		input_args[i].nr_rows = max_rows_per_dpu;
	}
	//printf("max:%d , prev:%d\n",max_rows_per_dpu,prev_rows_dpu);
	A = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
	B = malloc(n_size_pad * sizeof(T));
	C = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));

	// Initialize data with arbitrary data
	init_data(A, B, m_size, n_size);

	// Timer
	Timer timer;
	Timer1 timer1;
	
	/*********************************verification************************************/
	uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	struct AES_ctx ctx;
	float sec=0;
	int k=0;
#if VERIF
	uint8_t s1[]={ 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
    	
    	
   	//  s1[0]=(uint8_t)(A);
    T *firstTag1 = malloc((n_size)*sizeof(T)); 

	for (int i = 0; i < (n_size); i++){
	        firstTag1[i] =0; 
    	}

	int seed = 2; //(uint8_t)(A);

	startTimer(&timer1);
   	AES_init_ctx(&ctx, key);
   	AES_ECB_encrypt(&ctx, s1);

	int pow = 1;
   	for (int i = 0; i < (n_size*m_size); i++){ 	
		firstTag1[i%(n_size)] += (A[i]* (pow));
		if( pow == 64){
			pow = 1;
		}
		else pow *= seed; 
        
   	}

    for (unsigned int i = (m_size * n_size)-n_size; i < (m_size * n_size); i++)
	{
		A[i] = firstTag1[k];
		k++;
		
	}
	stopTimer(&timer1);
	sec = getElapsedTime(timer1);
#endif
	/*********************************verification************************************/
	
	
	/*********************************SecNDP ***************************************/
	
	
	uint8_t* first = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(uint8_t));
    	T* ciphertext = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
    	T* temp = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
    	T* C_host = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
    	
    	startTimer(&timer1);
    	for(int i=0;i< max_rows_per_dpu * nr_of_dpus * n_size_pad; i++){

		 first[i] = (uint8_t)(A + sizeof(uint32_t));
        //printf("%d  \n", ciphertext[i]);
    	}
    	AES_init_ctx(&ctx, key);
    	AES_ECB_encrypt(&ctx, first);
    	for(int i=0;i< max_rows_per_dpu * nr_of_dpus * n_size_pad; i++){

        	ciphertext[i] = (uint32_t)first[i];
			temp[i]= (A[i]) - ciphertext[i];	
     	}
     	
	printf("\n");
     	stopTimer(&timer1);
    	sec += getElapsedTime(timer1);
	/*********************************SecNDP****************************************/
	
	
	/*********************************integrity*************************************/
#if INTG	
	uint8_t s= 0x2b;
    	startTimer(&timer1);

    T *firstTag = malloc((m_size)*sizeof(T)); 

	for (int i = 0; i < (m_size); i++){
	        firstTag[i] =0; 
    	}
   	 
   	for (int i = 0; i < (n_size*m_size); i++){
   		if(i%(n_size)!=(n_size-1)){
    		T mul = (ciphertext[i]) * (s);
    		
       	 firstTag[i/(n_size)] += mul; 
       	 //printf("s:%u cipher: %u, tag:%u mul:%u\n",s, ciphertext[i], firstTag[i/(n_size)],mul);
        }
   	 }
    	int countTags=0;

     k=-1;
    for (unsigned int i = 0; i < max_rows_per_dpu * nr_of_dpus * n_size_pad; i++)
	{
	    if(i%n_size == n_size-1){
	    	k++;
		ciphertext[i] = firstTag[k];
		
		}
	if(i > (m_size * n_size)-1)  ciphertext[i]=0;	
	
	if(i ==(max_rows_per_dpu * nr_of_dpus * n_size_pad)-1) ciphertext[i] = firstTag[k];
	}
	stopTimer(&timer1);
    	 sec += getElapsedTime(timer1);
   	
 #endif 
	/*********************************integrity*************************************/
	
	
	// Compute output on CPU (performance comparison and verification purposes)
	start(&timer, 0, 0);
	gemv_host(C, A, B, m_size, n_size);
	stop(&timer, 0);
	float cpu=0;
	//printf("row pwr dpu%d    %d\n",max_rows_per_dpu,n_size_pad); 
	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {



		if (rep >= p.n_warmup)
			start(&timer, 1, rep - p.n_warmup);
		// Input arguments
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			// Copy input arguments to DPU
			input_args[i].max_rows = max_rows_per_dpu;

			DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
		}

		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));

		// Copy input array and vector
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, ciphertext + dpu_info[i].prev_rows_dpu * n_size));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * (n_size_pad) * sizeof(T), DPU_XFER_DEFAULT));
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, B));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * (n_size_pad) * sizeof(T) , n_size_pad * sizeof(T), DPU_XFER_DEFAULT));

		if (rep >= p.n_warmup)
			stop(&timer, 1);

		// Run kernel on DPUs
		if (rep >= p.n_warmup)
		{
			start(&timer, 2, rep - p.n_warmup);
#if ENERGY
			DPU_ASSERT(dpu_probe_start(&probe));
#endif
		}

		DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

		if (rep >= p.n_warmup)
		{
			stop(&timer, 2);
#if ENERGY
			DPU_ASSERT(dpu_probe_stop(&probe));
#endif
		}

		// Display DPU Logs
		DPU_FOREACH(dpu_set, dpu) {
			DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		}


		/**********************cpu part***************************/
		if (rep >= p.n_warmup) startTimer(&timer1);
		gemv_host(C_host, temp, B, m_size, n_size);

		if (rep >= p.n_warmup) stopTimer(&timer1);
 		if (rep >= p.n_warmup) cpu += getElapsedTime(timer1);
 		 //printf("temp cal done\n");
		/**********************cpu part***************************/
		
		
		// Retrieve results
		C_dpu = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
		if (rep >= p.n_warmup)
			start(&timer, 3, rep - p.n_warmup);
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu + i * max_rows_per_dpu));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
		if(rep >= p.n_warmup)
			stop(&timer, 3);
	}
#if ENERGY
	double acc_energy, avg_energy, acc_time, avg_time;
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_ACCUMULATE, &acc_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &avg_energy));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_ACCUMULATE, &acc_time));
	DPU_ASSERT(dpu_probe_get(&probe, DPU_TIME, DPU_AVERAGE, &avg_time));
#endif
	/*************************combine results*************************/
    startTimer(&timer1);
    T* C_total =  malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
    	
    for(int i=0; i<max_rows_per_dpu * nr_of_dpus ; i++){
    	C_total[i]= C_host[i] + C_dpu[i];
    	//printf("%u %u, %u, %u\n",C[i], C_total[i], C_host[i], C_dpu[i]);
    }

#if VERIF
	int32_t firstTag2 = 0;
	seed= 2;//(uint8_t)(A[NUM_LAYERS-1]);
	pow = 1;

	for (unsigned int i = 0; i < m_size-1; i++){
		int mul = ((C_total[i]) * (pow));
		firstTag2 += mul;
		if(pow == 64){ pow = 1;} 
		else pow *= seed;
	}
	int uplimit = C_total[m_size-1] + C_total[m_size-1]/100;
	int lowlimit = C_total[m_size-1] - C_total[m_size-1]/100;
	if (lowlimit <= firstTag2 && firstTag2 <= uplimit) printf("Verified!");
#endif	

	stopTimer(&timer1);
    float merge = getElapsedTime(timer1);
    
    /*************************combine results*************************/
    
	// Print timing results
	printf("\nCPU Version Time (ms): ");
	print(&timer, 0, 1);
	// printf("\nCPU-DPU Time (ms):");
	// print(&timer, 1, p.n_reps);
	// printf("\nDPU Kernel Time (ms):");
	// print(&timer, 2, p.n_reps);
	// printf("\nDPU-CPU Time (ms): ");
	// print(&timer, 3, p.n_reps);
	// printf("\nSec Cal Time(ms): %f   \nCPU Part Time(ms): %f   \nMerge Time(ms): %f  \n", sec*1e3, (cpu/p.n_reps)*1e3, merge*1e3);

	double cpuside = (cpu/p.n_reps)*1e3;
	double dpuside = ((timer.time[1]) + (timer.time[2]) + (timer.time[3])) / (1000*p.n_reps);
	double exec = fmax(cpuside,dpuside) +  merge*1e3;
	printf("\nTotal Execution time (ms): %f", exec);

#if ENERGY
	printf("Energy (J): %f J\t", avg_energy);
#endif

	// Check output
	bool status = true;
	unsigned int n,j;
	i = 0;
	for (n = 0; n < nr_of_dpus; n++) {
		for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
			if((n*dpu_info[n].rows_per_dpu)+j< m_size & C[i] != C_total[n * max_rows_per_dpu + j]) {
				status = false;
				printf(" %u, %u \n",C_total[n * max_rows_per_dpu + j],C[i]);
#if PRINT
				// printf("%d: %d -- %d\n", i, C[i], C_dpu[n * max_rows_per_dpu + j]);
#endif
			}
			// printf(" %u, %u \n",C_total[n * max_rows_per_dpu + j],C[i]);
			i++;
		}
	}
	if (status) {
		printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs and Tag are equal\n");
	} else {
		printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
	}

	// Deallocation
	free(A);
	free(B);
	free(C);
	free(C_dpu);
	free(C_host);
	free(C_total);
	free(temp);
	free(ciphertext);
	free(first);
#if VERIF 
	free(firstTag1);
#endif
#if INTG
	free(firstTag);
#endif
	DPU_ASSERT(dpu_free(dpu_set));

#if ENERGY
	DPU_ASSERT(dpu_probe_deinit(&probe));
#endif

	return status ? 0 : -1;
}
