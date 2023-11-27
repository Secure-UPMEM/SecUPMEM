 
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
	//printf("A\n");
	for(int lay = 0; lay < NUM_LAYERS; lay++){
		//printf("lay: %d\n", lay);
		#pragma omp parallel for
		for (unsigned int m = 0; m < m_size; m++) {
			for (unsigned int n = 0; n < n_size; n++)
			{
				//if(m %100 <98
				A[lay][m * n_size + n] = (unsigned int) (rand()%50);//1.1;//
				//printf("hi ");
			}
		//printf("\n");
		}
	}
	srand(0);
	//printf("B\n");
	for(int b = 0; b < batch ; b++){
		#pragma omp parallel for
		for (unsigned int i = 0; i < n_size; i++)
		{
			B[b][i] = (unsigned int) (rand()%25);
		//printf("%f  ", B[i]);
		}
	}
		//B[n_size-1] = 0;
}



void gemv(T** A, T* B, unsigned int m_size, unsigned int n_size, T* C, int lay) {//may change
	/*#pragma omp parallel for
	for (unsigned int i = 0; i < m_size; i++)
	{
		C[i] = 0;
	}*/
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
	/*#pragma omp parallel for
	for (unsigned int i = 0; i < m_size; i++)
	{
		C[i] = 0;
	}*/
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
	int max_batch =2560;
	int min_batch= 1;
#if VERIF
	//m_size++;// increase the matrix by one for verification tags
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
	//printf("start\n");
	//printf("NR_DPUS:%d\n",NR_DPUS);
	//printf("NUM_LAYERS:%d\n",NUM_LAYERS);
	//printf("group:%d\n",(NR_DPUS * 64000000));
	uint64_t inputsize= n_size * nr_of_dpus*sizeof(T);
	uint64_t outputsize= m_size * nr_of_dpus*sizeof(T);
	uint64_t mat= n_size*m_size;
	uint64_t numsize= NUM_LAYERS * sizeof(T);
	uint64_t totmat = mat * numsize;
	uint64_t matrixsize= totmat;
	uint64_t mram_sie=64000000;
	printf("input:%lu\n", inputsize);
	printf("input:%lu\n", outputsize);
	printf("matrix:%lu\n", matrixsize);
	printf("TAGHSIM:%lu\n", inputsize+outputsize+matrixsize);
	uint64_t bala = mram_sie * NR_DPUS;
	printf("nr_dpus:%lu\n", NR_DPUS );
	printf("bala: %lu\n", bala);
	group_number= (uint64_t)((bala) / (inputsize+outputsize+matrixsize));
	if (group_number >= batch_size) group_number = batch_size; 
	i = 0;
	printf("group:%d\n",group_number);
	uint32_t nr_dpus_group = nr_of_dpus/group_number;
	//printf("ok\n");
	DPU_FOREACH(dpu_set, dpu, i) {
		//printf("ok1\n");
		uint32_t rows_per_dpu;
		uint32_t prev_rows_dpu = 0;
		//uint32_t chunks = m_size / nr_of_dpus;

		uint32_t chunks = (uint32_t)((float)m_size / (float)nr_dpus_group);
		rows_per_dpu = chunks;
		uint32_t rest_rows = m_size % nr_dpus_group;
		//printf("ok2\n");
		if (i < rest_rows)
			rows_per_dpu++;
		//printf("ok3\n");
		if (rest_rows > 0) {
			if (i >= rest_rows)
				prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
			else
				prev_rows_dpu = i * (chunks + 1);
		} else {
			prev_rows_dpu = i * chunks;
		}
		//printf("ok4\n");
		// Keep max rows for parallel transfers
		uint32_t rows_per_dpu_pad = rows_per_dpu;
		if (rows_per_dpu_pad % 2 == 1) // 4-byte elements
			rows_per_dpu_pad++;
		if (rows_per_dpu_pad > max_rows_per_dpu)
			max_rows_per_dpu = rows_per_dpu_pad;
		//printf("ok5\n");
		dpu_info[i].rows_per_dpu = max_rows_per_dpu;//rows_per_dpu;
		dpu_info[i].rows_per_dpu_pad = max_rows_per_dpu;//rows_per_dpu_pad; //may change
		dpu_info[i].prev_rows_dpu = prev_rows_dpu;
		//printf("max:%d , prev:%d\n",max_rows_per_dpu,prev_rows_dpu);
		// Copy input arguments to DPU
		//printf("ok6\n");
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
	for(int b=0; b< batch_size; b++){
		B[b] = malloc(n_size_pad * sizeof(T));
	}
	//B = malloc(n_size_pad * sizeof(T));
	//B_temp = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	C = malloc(batch_size * sizeof(T*));
	for(int b=0; b< batch_size; b++){
		C[b] = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	}
	//C = malloc(group_number*max_rows_per_dpu * nr_of_dpus * sizeof(T));

	// Initialize data with arbitrary data
	init_data(A, B, m_size, n_size, batch_size);
	printf("initialization done\n");
	// Timer
	Timer timer;
	Timer1 timer1;
	printf("num inputs:%d, groups of parallel computation:%d\n",batch_size,group_number);
	/*********************************verification************************************/
	uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	struct AES_ctx ctx;
	float sec=0;
	int k=0;
	uint8_t s1[]={ 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
#if VERIF
    printf("1\n");
    T **firstTag1 = malloc((NUM_LAYERS)*sizeof(T*)); 
    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
   	 firstTag1[lay] = malloc((n_size)*sizeof(T)); 
    }
	printf("1\n");
   T **verifTag = malloc((NUM_LAYERS)*sizeof(T*)); 
    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
         verifTag[lay] = malloc((batch_size)*sizeof(T)); 
    }
	printf("1\n");/*
    uint8_t **totTag1 = malloc((NUM_LAYERS)*sizeof(uint8_t*)); 
    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
         totTag1[lay] = malloc((n_size)*sizeof(uint8_t)); 
    }*/
	printf("1\n");
    //T firstTag1 =0;
    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
	#pragma omp parallel for
	  for (int i = 0; i < (n_size); i++){
	        firstTag1[lay][i] =0; 
    	}
    }
        //      printf("sec\n");
                //AES_init_ctx(&ctx, key);
                //AES_ECB_encrypt(&ctx, first);

    startTimer(&timer1);
	printf("1\n");
    for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
	//s1[0]=(uint8_t)(B);
	    s1[0]=(uint8_t)(A[lay]);
		//printf("2\n");
	 //   #pragma omp parallel for
   	    for (int i = 0; i < (m_size*n_size); i++){ 
		//totTag1[i] = (uint8_t)(B+sizeof(T));
		//printf("no\n");
		T mul = pow(s1[0],((n_size)-(i%n_size))) * A[lay][i];
		//printf("no1\n");
       	 	firstTag1[lay][i%(n_size)] += mul;        
		//printf("no2\n");
   	    }
		//printf("3\n");
	//#pragma omp parallel for
	for(int b=0; b< batch_size; b++){
		//printf("ok\n");
		//#pragma omp parallel for
            for (int i = 0; i < (n_size); i++){
                verifTag[lay][b] += (firstTag1[lay][i] * B[b][i]);
            }
		//printf("oktar\n");
	}
	//printf("1\n");
   	 //firstTag1 += B[i];
	    /*#pragma omp parallel for
            for (int i = 0; i < (n_size); i++){
		totTag1[lay][i] = (uint8_t)(A[lay]);
 	    }
printf("before\n");
	    AES_init_ctx(&ctx, key);
            AES_ECB_encrypt(&ctx, totTag1[lay]);
	k=0;
	printf("after\n");
	#pragma omp parallel for
         for (unsigned int i = (m_size * n_size); i < (m_size * n_size); i++)
	 {
	 	A[lay][i] = firstTag1[lay][k]-totTag1[lay][k];
		k++;

	 }*/
     }
	//printf("1\n");
	 //B[n_size] = firstTag1;
     stopTimer(&timer1);
     sec = getElapsedTime(timer1);

#endif
	/*********************************verification************************************/
	
	
	/*********************************SecNDP ***************************************/
	printf("sec\n");
	uint8_t* first;
	//first = malloc(group_number * sizeof(uint8_t*));
	//for(int b=0; b< group_number; b++){
		first = malloc(n_size_pad * sizeof(uint8_t));
	//}
	printf("sec2\n");
	T** ciphertext;
	ciphertext = malloc(batch_size * sizeof(T*));
	for(int b=0; b< batch_size; b++){
		ciphertext[b] = malloc(n_size_pad * sizeof(T));
	}
	printf("sec3\n");
	T** ciphertexttemp;
	ciphertexttemp = malloc(batch_size * sizeof(T*));
	for(int b=0; b< batch_size; b++){
		ciphertexttemp[b] = malloc(n_size_pad * sizeof(T));
	}
	printf("sec4\n");
	T1* temp = malloc(n_size_pad * sizeof(T));
	printf("sec5\n");
    	T * C_host;
    	 
		C_host = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	
	printf("sec6\n");
    	startTimer(&timer1);
    	for(int b=0; b< batch_size; b++){
		#pragma omp parallel for
   		for(int i=0;i< n_size_pad; i++){
			first[i] = (uint8_t)(B+sizeof(T));
    		}
    	//	printf("sec\n");
        	AES_init_ctx(&ctx, key);
    		AES_ECB_encrypt(&ctx, first);
    	//	printf("sec\n");
		#pragma omp parallel for
    		for(int i=0;i<n_size_pad; i++){

	        	temp[i] = (T1)first[i];
			ciphertext[b][i]= B[b][i] - temp[i];
			ciphertexttemp[b][i]=ciphertext[b][i];
	     	}
	  //   	printf("sec\n");
	}  
	printf("sec done\n");
     	stopTimer(&timer1);
    	sec += getElapsedTime(timer1);
	/*********************************SecNDP****************************************/
	
	
	/*********************************integrity*************************************/
#if INTG	
/*	uint8_t s= 0x2b;
    	startTimer(&timer1);

    T *firstTag = malloc((m_size)*sizeof(T1)); 

	for( uint64_t lay=0; lay < NUM_LAYERS; lay++){
		k=0;
		for (int i = 0; i < (m_size); i++){
		        firstTag[i] =0; 
    		}
   	 
   		for (int i = 0; i < (n_size*m_size); i++){
   			if(i%(n_size)!=(n_size-1)){
   	 			T mul = (ciphertext[lay][i]) * (s); 	
	       		 firstTag[i/(n_size)] += mul; 
       	 //printf("s:%u cipher: %u, tag:%u mul:%u\n",s, ciphertext[i], firstTag[i/(n_size)],mul);
        		}
   		 }
   	 //printf("ok \n");
   	 	int countTags=0;
	     k=-1;
	    for (unsigned int i = 0; i < max_rows_per_dpu * nr_of_dpus * n_size_pad; i++){
		    if(i%n_size == n_size-1){
	    			k++;
				ciphertext[lay][i] = firstTag[k];		
			}
		    if(i > (m_size * n_size)-1)  ciphertext[lay][i]=0;	
	
	            if(i ==(max_rows_per_dpu * nr_of_dpus * n_size_pad)-1) ciphertext[lay][i] = firstTag[k];
	    }
	}
	stopTimer(&timer1);
        sec += getElapsedTime(timer1);
     	//printf("\n");
    	//printf("integ done \n");
    	*/
   	
 #endif 
	/*********************************integrity*************************************/
	//C_dpu = malloc(group_size * max_rows_per_dpu * nr_of_dpus * sizeof(T));
	C_dpu = malloc(batch_size * sizeof(T*));
	for(int b=0; b< batch_size; b++){
		C_dpu[b] = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	}
	/*for(int i=0;i<10;i++){
		for(int j=0; j<1024*1024*1024; j++)
		flush[i]=i*j;
		
	}*/
	// Compute output on CPU (performance comparison and verification purposes)
	start(&timer, 0, 0);
	startTimer(&timer1);
	//gemv_host(C, A, B, m_size, n_size);
	for(int lay = 0; lay < NUM_LAYERS; lay++){
		for(int b=0 ; b< batch_size;b++){
			gemv(A, B[b%group_number], m_size, n_size,C[b%group_number], lay);
		for (unsigned int i = 0; i < m_size; i++)
		{
			if(C[b%group_number][i] <= 0) C[b%group_number][i] = 0;
		}
	}}
	stopTimer(&timer1);
        float cpuplain = getElapsedTime(timer1);

	stop(&timer, 0);
	float cpu[NUM_LAYERS]={0};
	float temp_gen[NUM_LAYERS]={0};
	float precal={0};
	float cpudpu[NUM_LAYERS]={0};
	float interdpu[NUM_LAYERS]={0};
	float kernel[NUM_LAYERS]={0};
	float dpucpu[NUM_LAYERS]={0};
	T1* temp2 = malloc(n_size_pad  * sizeof(T1));
	//T* C_total =  malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	T** C_total;
	C_total = malloc(batch_size * sizeof(T*));
	for(int b=0; b< batch_size; b++){
		C_total[b] = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
	}
uint64_t counter=0;
uint64_t counter2 =0;
	//printf("row pwr dpu\n"); 
	uint8_t* first2 = malloc(n_size_pad * sizeof(uint8_t));
	/*for (unsigned int i = 0; i < m_size; i++)
			{
				C_host[i] = 0;
			}*/
	printf("send info\n");
	
	for(int lay=0; lay <  NUM_LAYERS; lay++){
		
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, A[lay] + ((dpu_info[i].prev_rows_dpu * n_size))));//%(n_size*m_size))));
		}
		printf("send weights: %d\n", lay);
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * (n_size_pad) * sizeof(T), DPU_XFER_DEFAULT));//lay * max_rows_per_dpu * (n_size_pad) * sizeof(T), max_rows_per_dpu * (n_size_pad) * sizeof(T), DPU_XFER_DEFAULT));
	}
		printf("send weights\n");
		
	//}
	printf("done send weights\n");
	for (unsigned int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
		printf("%d\n",batch_size);
	    //for(int inputs=0; inputs< batch_size; inputs += group_number){	
		i = 0;
		if (rep >= p.n_warmup)
			start(&timer, 1, rep - p.n_warmup);
		// Input arguments
		if (rep >= p.n_warmup)
                {
			startTimer(&timer1);
		}
		printf("start\n");
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			// Copy input arguments to DPU
			input_args[i].max_rows = max_rows_per_dpu;

			DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
		}
		printf("send args\n");
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), DPU_XFER_DEFAULT));
		printf("sending layer\n");
		//send current layer
		DPU_FOREACH(dpu_set, dpu, i) {
			int32_t lay1 = 0;		
			DPU_ASSERT(dpu_prepare_xfer(dpu, &lay1));
		}
		printf("sending layer1\n");
		//printf("sending vector and matrix\n");
		// Copy input array and vector
		
		printf("send inputs\n");
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, ciphertext[(i%group_number)]));// ciphertext here is the B which is the vector
			//printf("i: %d\n", i);
		}
		printf("send inputs\n");
		//printf("group:%d\n", group_number);
		//printf("size of ciphrt:%d\n", sizeof(ciphertext));
		//printf("n_size_pad:%d\n", n_size_pad);
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, NUM_LAYERS * (max_rows_per_dpu * (n_size_pad) * sizeof(T)) , n_size_pad * sizeof(T), DPU_XFER_DEFAULT));
		
		// copy weights
		
		if (rep >= p.n_warmup){

		stopTimer(&timer1);
		cpudpu[0] += getElapsedTime(timer1);
		}
		if (rep >= p.n_warmup)
			stop(&timer, 1);

		/********************************add cpu portion ***********************************/
		//uint8_t* first2 = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(uint8_t));
		printf("cpu start\n");
		/*add  verification related stuff*/

		/* end verif stuff*/

		if (rep >= p.n_warmup) startTimer(&timer1);
//		for(int b=0;b< group_size; b++){
			#pragma omp parallel for
			for(int i=0;i< (n_size_pad); i++){
				 first2[i] = (uint8_t)(B+sizeof(T));
	    		}
	    		AES_init_ctx(&ctx, key);
	    		AES_ECB_encrypt(&ctx, first2); 
			gemv1(A[0],first2, m_size, n_size, C_host);
//		}
		counter2++;
		if (rep >= p.n_warmup) stopTimer(&timer1);
 		if (rep >= p.n_warmup && counter2 <= group_number) cpu[0] += getElapsedTime(timer1);
		
 		printf("cpu cal done\n");
		
		/******************************** end of added cpu portion ************************/
		
		
		// Run kernel on DPUs
		if (rep >= p.n_warmup)
		{
			start(&timer, 2, rep - p.n_warmup);
#if ENERGY
			DPU_ASSERT(dpu_probe_start(&probe));
#endif
		}
		if (rep >= p.n_warmup)
                {
                        startTimer(&timer1);
                }

		DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
		if (rep >= p.n_warmup){
		stopTimer(&timer1);
                kernel[0] += getElapsedTime(timer1);
		}
		if (rep >= p.n_warmup)
		{
			stop(&timer, 2);
#if ENERGY
			DPU_ASSERT(dpu_probe_stop(&probe));
#endif
		}

		//START LAYER 2 AND BEYOND
		
		for(int lay = 1; lay < NUM_LAYERS; lay++){
			if (rep >= p.n_warmup)
				start(&timer, 4, rep - p.n_warmup);
			if (rep >= p.n_warmup)
                {
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
		for(int i=0; i<max_rows_per_dpu * nr_of_dpus/partition ; i++){
		 //#pragma omp parallel for
		for(int b=0; b< group_number;b++){
		 //#pragma omp parallel for
			for(int part= i*100; part<= (i+1)*100; part++){
 //   		    for(int i=0; i<max_rows_per_dpu * nr_of_dpus/part ; i++){
		//	C_dpu = C_total[i];    	
    			if( C_host[part] + C_dpu[b][part] > 0)
    				C_total[b][part]= C_host[part] + C_dpu[b][part];
    			else
    				C_total[b][part]=0;
    			//printf("hi\n");
    		   }
    		}
		}
			if (rep >= p.n_warmup){

			stopTimer(&timer1);
			interdpu[lay-1] += getElapsedTime(timer1);
    		   printf("combine\n");
			}
    
		    /*************************combine results*************************/
			// B = C
			/*unsigned int n, j;
			i = 0;
			for (n = 0; n < nr_of_dpus; n++) {
				for (j = 0; j < dpu_info[n].rows_per_dpu; j++) {
					B_tmp[i] = C_dpu[n * max_rows_per_dpu + j];
					i++;
				}
			}*/
			i = 0;
			if (rep >= p.n_warmup)
                {
                        startTimer(&timer1);
                }
                //send current layer
		DPU_FOREACH(dpu_set, dpu, i) {
			int32_t lay1 = lay;		
			DPU_ASSERT(dpu_prepare_xfer(dpu, &lay1));
		}
		//printf("sending layer2\n");
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "CURRENT_LAYER", 0, sizeof(int32_t), DPU_XFER_DEFAULT));
		// I need to add privacy here.
			DPU_FOREACH(dpu_set, dpu, i) {
				DPU_ASSERT(dpu_prepare_xfer(dpu, C_total[(i%group_number)]));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, NUM_LAYERS * (max_rows_per_dpu * (n_size_pad) * sizeof(T)) , n_size_pad * sizeof(T), DPU_XFER_DEFAULT));

			// Copy next matrix of weights
			i = 0;
			/*DPU_FOREACH(dpu_set, dpu, i) {
				DPU_ASSERT(dpu_prepare_xfer(dpu, A[lay] + dpu_info[i].prev_rows_dpu * n_size));
			}
			DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT));*/
			if (rep >= p.n_warmup){

			stopTimer(&timer1);
			cpudpu[lay] += getElapsedTime(timer1);
			}
			if(rep >= p.n_warmup)
				stop(&timer, 4);
			
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
		//uint8_t* first2 = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(uint8_t));
		printf("cpu start\n");
		if (rep >= p.n_warmup) startTimer(&timer1);
//		for(int b=0;b< group_size; b++){
			#pragma omp parallel for
			for(int i=0;i< (n_size_pad); i++){
				 first2[i] = (uint8_t)(B+sizeof(T));
	    		}
	    		AES_init_ctx(&ctx, key);
	    		AES_ECB_encrypt(&ctx, first2); 
			gemv1(A[lay],first2, m_size, n_size, C_host);
//		}
		counter++;
		if (rep >= p.n_warmup) stopTimer(&timer1);
 		if (rep >= p.n_warmup && counter <= group_number) cpu[lay] += getElapsedTime(timer1);

 		printf("cpu cal done\n");
		
		/******************************** end of added cpu portion ************************/
		/**********************cpu part***************************/
		}	
				
				
		// Retrieve results
		printf("retrived\n");
		if (rep >= p.n_warmup)
			start(&timer, 3, rep - p.n_warmup);
		i = 0;
		DPU_FOREACH(dpu_set, dpu, i) {
			DPU_ASSERT(dpu_prepare_xfer(dpu, C_dpu[(i%group_number)] + i * max_rows_per_dpu));
		}
		DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, max_rows_per_dpu * n_size_pad * sizeof(T) + n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));
		if(rep >= p.n_warmup)
			stop(&timer, 3);
		
		// Display DPU Logs
		DPU_FOREACH(dpu_set, dpu) {
			DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
		}
		for(int b=0; b< group_number; b++){
		for(int i=0;i<  n_size_pad; i++){        	
			ciphertext[b][i]=ciphertexttemp[b][i];
		}
		}
	//}
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
    printf("ok\n");
    //T1* C_total =  malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T1));
    //printf("lets combine\n");	
    //for(int b=0; b< group_number;b++){
		//printf("bye\n");
		//printf("bye\n");
		//printf("bye\n");
            /*     #pragma omp parallel for
                    for(unsigned int i=0; i<max_rows_per_dpu * nr_of_dpus ; i++){
	              // printf("ok\n");
      C_dpu = C_total[i];     
                        if( C_host[i] + C_dpu[b][i] > 0)
                                C_total[b][i]= C_host[i] + C_dpu[b][i];
                        else
                                C_total[b][i]=0;
                        //printf("hi\n");
                   }
                }*/
    printf("Combine done\n");
    stopTimer(&timer1);
    float merge = getElapsedTime(timer1);
    
    /*************************combine results*************************/
    /************************verify*********************************/
    float cpuon =0;
    float verif =0;
    float verift =0;
    s1[0]=(uint8_t)(A[NUM_LAYERS-1]);
    //T1 verift=0;
    T **firstTag2 = malloc((NUM_LAYERS)*sizeof(T*)); 
    for(int lay=0 ;lay< NUM_LAYERS; lay++){
	firstTag2[lay] = malloc(batch_size* sizeof(T));
	}
	//float verif =0;
#if VERIF
    
	startTimer(&timer1);
   	//AES_init_ctx(&ctx, key);
   	//AES_ECB_encrypt(&ctx, s1);
	//printf("hi\n");
	//for(int lay=0; lay< NUM_LAYERS; lay++){ 
		//printf("first in\n");
		for(int b=0; b< batch_size; b++){
	//		printf("second in\n");
			firstTag2[NUM_LAYERS-1][b]=0;
			T pow = 1;
			T mul =1;
			// #pragma omp parallel for
   			//for(int b=0; b< group_number; b++){
			for (int i = 0; i < n_size; i++){
    	//			printf("third\n");
				pow *= s1[0];
				firstTag2[NUM_LAYERS-1][b] += (C_total[b][i]) * pow;
    		   		 //firstTag1[i%(n_size)] += mul; 
       			 	// firstTag2[NUM_LAYERS-1][b] += mul;//A[NUM_LAYERS-1][i]; 
       				 //firstTag1[((n_size*m_size)-i-1)%(n_size)] += A[lay][(n_size*m_size)-i-1];
        
   		 	}
	//		printf("bye\n");	
   			// #pragma omp parallel for
 			//	for(int b=0; b< group_number; b++){
			if (verifTag[NUM_LAYERS-1][b]==firstTag2[NUM_LAYERS-1][b]);
			verift = 0;
			//for (int i = 0; i < (n_size); i++){
			  //      verift += firstTag2[i]*C_total[b][i]; 

		    	//}
	///	printf("still\n");
		}
//	printf("done\n");
	//}
	stopTimer(&timer1);
	verif = getElapsedTime(timer1);
#endif		
	/***************************edn verification*********************/
    /*if(C_total[m_size-1]==verift){ printf("tags verified\n");}
    else printf("not verified\n");
    printf("verift= %f, c_totvERIF=%f \n",verift, C_total[m_size-1]);*/
//#endif   
  /*  startTimer(&timer1);
    T1* ciphertext1 = malloc( n_size_pad * sizeof(T1));
    T* decrypted = malloc(n_size_pad * sizeof(T));
    uint8_t* first1 = malloc( n_size_pad * sizeof(uint8_t));
    T* C_cputot =  malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
    //printf("cpu onlu start\n");
	
    for(int lay = 0; lay < NUM_LAYERS; lay++){
    	#pragma omp parallel for
    	for(int i=0;i<  n_size_pad; i++){
    	
    		first1[i]=(uint8_t)(B + sizeof(T1));

		 
    	}
    	
    //printf("cpu onlu cipher\n");
    	AES_init_ctx(&ctx, key);
    	AES_ECB_encrypt(&ctx, first1);
	
	#pragma omp parallel for
	for(int i=0;i<  n_size_pad; i++){

        	ciphertext1[i] = (T1)first1[i];
        	decrypted[i] =  ciphertext1[i] - ciphertext[i];
     	}

//printf("cpu onlu decript done\n");
	//gemv_host(C_cputot, ciphertext1, B, m_size, n_size);
	//printf("cpu onlu start\n");
	gemv1(A[lay], decrypted, m_size, n_size, C_cputot);
	
	for (unsigned int i = 0; i < m_size; i++)
	{
		if(C_cputot[i] <= 0) C_cputot[i] = 0;
	}
    }
	stopTimer(&timer1);
        cpuon = getElapsedTime(timer1);
	 verift=0;*/
     /*firstTag2 = malloc((n_size)*sizeof(double)); 

	for (int i = 0; i < (n_size); i++){
	        firstTag2[i] =0; 
    	}*/

#if VERIF
	//printf("1\n");
	/*#pragma omp parallel for
   	for (int i = 0; i < (n_size*m_size)-n_size-1; i++){
    		//T mul = (A[i]) * pow(s1[0],((i/(n_size))%10));
       	 //firstTag1[i%(n_size)] += mul; 
       	 firstTag2[i%(n_size)] += A[NUM_LAYERS-1][i]; 
       	 //firstTag1[((n_size*m_size)-i-1)%(n_size)] += A[(n_size*m_size)-i-1];
        
   	 }
   	 //printf("1\n");
   	 #pragma omp parallel for
 	for (int i = 0; i < (n_size); i++){
	        verift += firstTag2[i]*C_cputot[i]; 
    	}*/
#endif
//printf("2\n");
	/*if(C_cputot[m_size-1]==verift){};
    }
	stopTimer(&timer1);
		if(C_cputot[m_size-1]==verift){ printf("tags verified\n");}
		
		else printf("not verified\n");
		printf("verift= %f, c_totvERIF=%f \n",verift, C_cputot[m_size-1]);
		*/
	//stopTimer(&timer1);
	//printf("cpu onlu done\n");
    //float cpuon = getElapsedTime(timer1);
	// Print timing results
	printf("\nCPU only Time (ms): %f\n",cpuon*1e3);
	print(&timer, 0, 1);
	printf("\nCPU Version Time (ms): ");
	print(&timer, 0, 1);
	printf(" = %f", (cpuplain)*1e3);

	printf("\nCPU-DPU Time (ms):");
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

	printf("Inter-DPU Time (ms): ");
	print(&timer, 4, p.n_reps);
	for(int i=0; i< NUM_LAYERS ; i++){
                printf("\n lay[%d] = %f:",i, (interdpu[i]/p.n_reps)*1e3);
    	}
	printf("\nSec Cal Time(ms): %f \nMerge Time(ms): %f\n  verification time: %f\n", sec*1e3, merge*1e3, verif*1e3);
	for(int i=0; i< NUM_LAYERS ; i++){
                printf("\n CPU_Add_Gen_lay[%d] = %f:",i, (temp_gen[i]/p.n_reps)*1e3);
        }
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
