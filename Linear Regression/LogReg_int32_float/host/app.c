/**
* app.c
* LRGD_1.5 Host Application Source File
* int32 and float done  
* quantize the input data 
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
#include <time.h>
#include <math.h>
#include <omp.h>
#include <gmp.h>

#include "../support/common.h"
#include "../support/timer.h"
#include "../support/params.h"
#include "../support/aes.h"
// Define the DPU Binary path as DPU_BINARY here
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

#define MAXCHAR 500 

#if ENERGY
#include <dpu_probe.h>
#endif

//you need to find the best partitioning for each input to make sure that the CPU wil not be the bottleneck
#define PART1 80
#define PART2 160
// Pointer declaration
static T* X;
static T* X_C;
static T* X_D;
static T* Y;
static T* W;
// static float* W;

// Create float input arrays 
static void read_input_float(float* X, float* Y, float* W, unsigned int m_size, unsigned int n_size) {
    srand(0);

    printf("Predefined weight: ");
    for (unsigned int w = 0; w < n_size; ++w) {
        W[w] = (T) (w+1); 
        // W[w] = (T) (rand()%(n_size*2)); 
        printf("%d, ", (int) W[w]); 
    }

    for (unsigned int i = 0; i < m_size * n_size; ++i) {
        // X[i] = (rand()%50) + ((float) (rand()%100))/100; 
        X[i] = ((float) (rand()%10000)) / 10000; 
    }

    for (unsigned int j = 0; j < m_size; ++j) {
        float tmp = 0; 
        for (unsigned int k = 0; k < n_size; ++k) {
            tmp += X[j*n_size + k] * W[k] + ((float) (rand()%300))/1000; 
        }
        Y[j] = tmp; 
    }
    printf("\nSuccessfully generate input data (float).\n");
}

// Create fixed-point input arrays 
static void read_input_fp(float* X, float* Y, T* X_fp, T* Y_fp, unsigned int m_size, unsigned int n_size) {
    for (unsigned int j = 0; j < m_size; ++j) {
        Y_fp[j] = Y[j] * (1 << SHIFT_AMOUNT); 
        for (unsigned int k = 0; k < n_size; ++k) {
            X_fp[j*n_size + k] = X[j*n_size + k] * (1 << SHIFT_AMOUNT); 
        }
    }
    printf("Successfully quantize input data.\n");
}
int power(int a, int n) {
  if(n == 0) {
    return 1;
  }
  if (n&1 == 0) {
    return power(a*a, n/2);
  }else{
    return a * power(a*a, n/2);
  }
}

#ifdef FLOAT // float 
// Train weight coefficients in the host 
static void GD_host(T* X, T* Y, T* W, uint32_t m_size, uint32_t n_size, uint32_t iter_time, float lr) {
    printf("-----Start traing at host, float-----\n");

    // init wirght with random value
    for (uint32_t n = 0; n < n_size; ++n)
        W[n] = (T) 1; 

    for (uint32_t i = 0; i < iter_time; ++i) {
        
        // calculate gradient 
        T* gradient_tmp = calloc(n_size, sizeof(T)); 

        for (uint32_t j = 0; j < m_size; ++j) {
            T dot_product = 0; 
            for (unsigned int k = 0; k < n_size; ++k) {
                dot_product += X[j*n_size + k] * W[k]; 
            }

            for (unsigned int l = 0; l < n_size; ++l) {
                gradient_tmp[l] -= X[j*n_size + l] * (Y[j] - dot_product) / m_size; // avoid overflow 
            }
        } // gradient done 
        
        // update weight
        for (uint32_t m = 0; m < n_size; ++m) {
            W[m] = W[m] - (gradient_tmp[m] * lr); 
        }
        free(gradient_tmp); 
    } // end iteration
}

#else // int 
// read data from SUSY.csv, -l 0.000001 
static int read_input_SUSY_fp(T* X, T* Y, unsigned int m_size, unsigned int n_size) {
    printf("Reading training dataset from file...\n"); 

    FILE* fp;
    char row[MAXCHAR];
    char* token;
    unsigned int m = 0, n = 0; 

    fp = fopen("/home/rain/桌面/ETH-SAFARI/SRC/SUSY.csv", "r"); // add file path 
    if (fp == NULL) {
        perror("Can't open file!");
        return(-1);
    } 

    while (fgets(row, MAXCHAR, fp)) {
        token = strtok(row, ",");
        X[m*n_size] = atof(token) * (1 << SHIFT_AMOUNT); 

        token = strtok(NULL, ",");
        Y[m] = atof(token) * (1 << SHIFT_AMOUNT); 

        n = 1; 
        token = strtok(NULL, ",");
        while (token != NULL) {
            X[m*n_size + n] = atof(token) * (1 << SHIFT_AMOUNT); 
            token = strtok(NULL, ","); 
            n++; 
        } 
        m++; 
    }
    fclose(fp); 
    printf("Successfully generate input data. m = %d\n", m); 
    if (m != m_size) {
        printf("Error: invalid input m_size!\n");
        return -1; 
    }
    return 0; 
}

// Train weight coefficients in the host
static void GD_host_fp(T* X, T* Y, T* W, T** y_expected,T** gd_expected, uint32_t m_size, uint32_t n_size, uint32_t iter_time, float lr) {
    printf("-----Start traing at host, int-----\n");

    // init weight with random value
    for (uint32_t n = 0; n < n_size; ++n){
        
        // W[n] = (T) 1; 
        W[n] = (T) (1 << SHIFT_AMOUNT);
    }

    for (uint32_t i = 0; i < iter_time; ++i) {
        // calculate gradient 
        T* gradient_tmp = calloc(n_size, sizeof(T)); 

        for (uint32_t j = 0; j < m_size; ++j) {
            T dot_product = 0; 
            for (unsigned int k = 0; k < n_size; ++k) {
                dot_product += X[j*n_size + k] * W[k]; 
            }
            y_expected[i][j] = dot_product;
            for (unsigned int l = 0; l < n_size; ++l) {
                // avoid overflow
                gradient_tmp[l] -= X[j*n_size + l] * (Y[j]-(dot_product>>
                    SHIFT_AMOUNT)) >> (SHIFT_AMOUNT + OVERFLOW_SHIFT); 
            }
            
        } // gradient done 
        
        // update weight
        for (uint32_t m = 0; m < n_size; ++m) {
            W[m] = W[m] - (gradient_tmp[m] * lr) / (m_size>>OVERFLOW_SHIFT); 
            gd_expected[i][m] = gradient_tmp[m];
        }

        free(gradient_tmp); 
    } // end iteration
}
#endif 
static void GD_host_fp_test(T* X, T* Y, T* dot_product,T* gd_expected, uint32_t m_size, uint32_t n_size) {
T* gradient_tmp = calloc(n_size, sizeof(T));
    for (uint32_t j = 0; j < m_size; ++j) {
         
            for (unsigned int l = 0; l < n_size; ++l) {
                // avoid overflow
                gradient_tmp[l] -= X[j*n_size + l] * (Y[j]-(dot_product[j]>>
                    SHIFT_AMOUNT)) >> (SHIFT_AMOUNT + OVERFLOW_SHIFT); 
            }
 // gradient done 
            
        }

for (unsigned int l = 0; l < n_size; ++l) {

            gd_expected[l] = gradient_tmp[l];
}
   free(gradient_tmp);      
}
static void init_argument_tasklet(uint32_t tasklet_id, uint32_t nr_rows, uint32_t* rows_per_tasklet, uint32_t* start_row){
    unsigned int element_per_cacheY = 8 >> DIV; 
    unsigned int chunks = nr_rows / (NR_TASKLETS * element_per_cacheY);
    unsigned int dbl_chunks = chunks * element_per_cacheY;  
    *rows_per_tasklet = dbl_chunks; // rows per tasklet is multiple of element_per_cacheY
    unsigned int rest_rows = nr_rows % (NR_TASKLETS * element_per_cacheY); 

    if ((tasklet_id * element_per_cacheY) < rest_rows)
        *rows_per_tasklet += element_per_cacheY;
    if (rest_rows > 0) {
        if ((tasklet_id * element_per_cacheY) >= rest_rows) {
            if ((rest_rows % element_per_cacheY) != 0)
                *start_row = roundup(rest_rows, element_per_cacheY) + tasklet_id * dbl_chunks; 
            else
                *start_row = rest_rows + tasklet_id * dbl_chunks; 
        } else 
            *start_row = tasklet_id * (dbl_chunks + element_per_cacheY);
    } else {
        *start_row = tasklet_id * (dbl_chunks);
    }
 
}

static void compute_mae(const float* X, const float* Y, const float* W, int m_size, int n_size, 
    const char* comment) { 
    float reduction = 0; 
    float sum_of_Y = 0; 
    for (int m = 0; m < m_size; ++m) {
        float dot_product = 0.0;
        for (int n = 0; n < n_size; ++n) {
            dot_product += X[m*n_size + n] * W[n]; 
        }
        reduction += (fabsf(Y[m] - dot_product)) / m_size; 
        sum_of_Y += fabs(Y[m]) / m_size; 
    }
    // float mae = (float) reduction / m_size; 
    printf("MAE on %s = %.4f, avg Y = %.4f, error rate = %.2f%%\n", comment, reduction, sum_of_Y, \
        (reduction/sum_of_Y)*100); 
}

// Main of the Host Application
int main(int argc, char **argv) {
    
    struct Params p = input_params(argc, argv);
    struct dpu_set_t dpu_set, dpu;
    uint32_t nr_of_dpus;

#if ENERGY
    struct dpu_probe_t probe;
    DPU_ASSERT(dpu_probe_init("energy_probe", &probe));
#endif

    // Allocate DPUs and load binary
    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set)); 
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));
    DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
    printf("Allocated %d DPU(s)\n", nr_of_dpus);

    unsigned int i = 0;

    unsigned int iter_time = p.iter_time;
    float learning_rate = p.learning_rate; 

    unsigned int m_size = p.m_size;
    unsigned int n_size = p.n_size;

    printf("i = %d, lr = %.4f, m = %d, n = %d\n", iter_time, learning_rate, m_size, n_size); 

    // Initialize help data
    dpu_info = (struct dpu_info_t *) malloc(nr_of_dpus * sizeof(struct dpu_info_t));
    dpu_arguments_t *input_args = (dpu_arguments_t *) malloc(nr_of_dpus * sizeof(dpu_arguments_t));
    uint32_t max_rows_per_dpu = 0;
    uint32_t n_size_pad = ((n_size*sizeof(T)) % 8) == 0 ? n_size : roundup(n_size, (8/sizeof(T))); 

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
                prev_rows_dpu = rest_rows + i * chunks; 
                // prev_rows_dpu = rest_rows * (chunks + 1) + (i - rest_rows) * chunks;
            else
                prev_rows_dpu = i * (chunks + 1);
        } else {
            prev_rows_dpu = i * chunks;
        }

        // Keep max rows for parallel transfers
        uint32_t rows_per_dpu_pad = ((rows_per_dpu*sizeof(T)) % 8) == 0 ? rows_per_dpu : roundup(rows_per_dpu, (8/sizeof(T))); 
        if (rows_per_dpu_pad > max_rows_per_dpu)
            max_rows_per_dpu = rows_per_dpu_pad;

        dpu_info[i].rows_per_dpu = rows_per_dpu;
        dpu_info[i].rows_per_dpu_pad = rows_per_dpu_pad;
        dpu_info[i].prev_rows_dpu = prev_rows_dpu;

        // Copy input arguments to DPU
        input_args[i].n_size = n_size;
        input_args[i].n_size_pad = n_size_pad;
        input_args[i].nr_rows = rows_per_dpu;

        // Init arguments for each tasklet
        for(uint32_t id = 0; id < NR_TASKLETS; ++id) {
            init_argument_tasklet(id, rows_per_dpu, &input_args[i].rows_per_tasklet[id], \
                &input_args[i].start_row[id]); 
        }
    }

    // Input/output allocation
    X = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T));
    X_C = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T)); 
    X_D = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(T)); 
    Y = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
    T** y_expected = malloc(iter_time * sizeof(T*)); 
    for(unsigned int i=0; i< iter_time; i++){
        y_expected[i] = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T));
    }
    T* Y_host = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T)); 
    T* Y_dpu = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T)); 
    T* Y_total = malloc(max_rows_per_dpu * nr_of_dpus * sizeof(T)); 
    W = malloc(n_size_pad * sizeof(T)); 

    // init trainging dataset and weight for host 
    T *bufferX = X;
    T *bufferY = Y;
    T *bufferW_host = W; 

    // init training dataset and initial host W 
    # ifdef FLOAT 
    read_input_float(bufferX, bufferY, bufferW_host, m_size, n_size);
    # else 
    float* X_float = (float*) malloc(m_size * n_size* sizeof(float)); 
    float* Y_float = (float*) malloc(m_size * sizeof(float)); 
    read_input_float(X_float, Y_float, bufferW_host, m_size, n_size); 
    read_input_fp(X_float, Y_float, bufferX, bufferY, m_size, n_size); 
    #endif

    // init Weight for DPU 
    T* W_dpu = malloc(n_size_pad * sizeof(T)); 
    T* W_dpu_fp = malloc(n_size_pad * sizeof(T)); 
    for (uint32_t n = 0; n < n_size_pad; ++n) {
        W_dpu[n] = (T) 1.0; 
        W_dpu_fp[n] = (T) (1 << SHIFT_AMOUNT); 
    }
    // temp dpu gradient  
    T* gradient_dpu_tmp = malloc(n_size_pad * nr_of_dpus * sizeof(T));

    T** gd_expected = malloc(iter_time * sizeof(T*)); 
    for(unsigned int i=0; i< iter_time; i++){
        gd_expected[i] = malloc(n_size * sizeof(T));
    }
    // Timer declaration
    Timer timer;

    // Train the model on host
    // clock_t start_cpu= clock();
    start(&timer, 0, 0);
    #ifdef FLOAT 
    GD_host(bufferX, bufferY, bufferW_host, m_size, n_size, iter_time, learning_rate); 
    #else 
    GD_host_fp(bufferX, bufferY, bufferW_host,y_expected,gd_expected, m_size, n_size, iter_time, learning_rate); 
    #endif 
    stop(&timer, 0);  

    //Generation ciphertext
    uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	struct AES_ctx ctx;
    uint8_t* counter = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad * sizeof(uint8_t));
    // #pragma omp parallel for
   	for(uint32_t i=0;i< max_rows_per_dpu * nr_of_dpus * n_size_pad; i++){
		counter[i] = (uint8_t)(0+(i*sizeof(T)));//(uint8_t)(bufferX+(i*sizeof(T)));
    }

    AES_init_ctx(&ctx, key);
    AES_ECB_encrypt(&ctx, counter);
	// #pragma omp parallel for
    for(uint32_t i=0;i<max_rows_per_dpu * nr_of_dpus * n_size_pad; i++){
        X_C[i] = (T)counter[i];
		X_D[i]= bufferX[i] - X_C[i];
	}
    for(uint32_t i=0;i<10; i++){
	}

    // Transfer input arguments and training dataset to DPU
    printf("Load input data to DPUs\n");
    start(&timer, 1, 0); // init CPU-DPU transfer time start
    // Input arguments
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        // Copy input arguments to DPU
        input_args[i].max_rows = max_rows_per_dpu;

        DPU_ASSERT(dpu_prepare_xfer(dpu, input_args + i));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0, sizeof(dpu_arguments_t), \
        DPU_XFER_DEFAULT)); 

    // Copy X and y 
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, X_D + dpu_info[i].prev_rows_dpu * n_size));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, \
        max_rows_per_dpu * n_size_pad * sizeof(T), DPU_XFER_DEFAULT)); 
    
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, bufferY + dpu_info[i].prev_rows_dpu)); 
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
        max_rows_per_dpu * n_size_pad * sizeof(T), max_rows_per_dpu * sizeof(T), DPU_XFER_DEFAULT));

    stop(&timer, 1); // init CPU-DPU transfer time stop
    uint32_t mode = 0;
    // Iteration at DPU
    printf("Run program on DPU(s)...\n"); 
    for(uint32_t rep = 0; rep < iter_time; ++rep) {
        // Copy W 
        start(&timer, 2, rep); // syn CPU-DPU transfer time start
        i = 0; 
        DPU_FOREACH(dpu_set, dpu, i) {
            #ifdef FLOAT
            DPU_ASSERT(dpu_prepare_xfer(dpu, W_dpu)); 
            #else
            DPU_ASSERT(dpu_prepare_xfer(dpu, W_dpu_fp)); 
            #endif 
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, \
            max_rows_per_dpu * n_size_pad * sizeof(T) + max_rows_per_dpu * sizeof(T), \
            n_size_pad * sizeof(T), DPU_XFER_DEFAULT)); 
        
        //mode send
        i = 0;
        mode = 0;
        DPU_FOREACH(dpu_set, dpu, i) {
        // Copy input arguments to DPU
            DPU_ASSERT(dpu_prepare_xfer(dpu, &mode));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "mode", 0, \
        sizeof(uint32_t), DPU_XFER_DEFAULT));
 
        //end mode

        stop(&timer, 2); // syn CPU-DPU transfer time stop 

        //runing the kernel on the CPU in parallel
        
        
        AES_init_ctx(&ctx, key);
        start(&timer, 6, rep);

        // // #pragma omp parallel for shared(W, counter1)
        // for (int s=0; s<PART ; s++){
        uint8_t* counter1 = malloc(max_rows_per_dpu * nr_of_dpus * n_size_pad/PART1 * sizeof(uint8_t));
        // #pragma omp parallel for shared(W, counter1)
        for (int s = 0; s < PART1; s++) {
            for (uint32_t i = 0; i < m_size/ PART1 ; i++) { //  
                int offset = s * (m_size / PART1) + i;
                for (unsigned int k = 0; k < n_size; k++) {
                    int local_off = i * n_size + k;
                    // counter[i] = (uint8_t)(0+(i*sizeof(T)));
                    counter1[local_off] = (uint8_t)(0+(((offset * n_size) + k) * sizeof(T)));//(uint8_t)(bufferX + (k * sizeof(T)));//[s*(max_rows_per_dpu * nr_of_dpus * n_size_pad/PART)+(i*(n_size_pad)+k)]);
                }
            }
            AES_ECB_encrypt(&ctx, counter1);

            for(uint32_t i=0;i< (m_size/PART1); i++){ 
                int offset = s * (m_size / PART1) + i;
                Y_host[offset]=0;
                for (unsigned int k = 0; k < n_size; k++) {
                    Y_host[offset] += counter1[(i*(n_size))+k] * W_dpu_fp[k]; 
                }
            }
        }
         stop(&timer, 6);

        // Run DPU kernel
        start(&timer, 3, rep); 
        #if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe)); 
        #endif
        // Launch kernel 
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS)); 
        stop(&timer, 3); 

        #if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
        #endif


#if PRINT
        {
            if (rep == 0) {
                unsigned int each_dpu = 0;
                printf("Display DPU Logs\n");
                DPU_FOREACH (dpu_set, dpu) {
                    if(each_dpu==0)printf("DPU#%d:\n", each_dpu);
                    DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                    each_dpu++;
                }
            }
        }
#endif
        start(&timer, 4, rep); // DPU-CPU time 
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, Y_dpu + dpu_info[i].prev_rows_dpu)); 
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "dot_product_t", 0, max_rows_per_dpu  * sizeof(T), \
            DPU_XFER_DEFAULT));
        stop(&timer, 4); // DPU-CPU time 
        start(&timer, 7, rep);
        
        for(uint32_t i=0;i< max_rows_per_dpu * nr_of_dpus ; i++){
		    Y_total[i] = Y_dpu[i] + Y_host[i];
        }
        // if(rep==1 || rep==0){
        //         for(int i= (max_rows_per_dpu * nr_of_dpus -10); i<max_rows_per_dpu * nr_of_dpus; i++){
        //             printf("y_d: %d, y_c: %d, y_t: %d, y_expexted: %d\n", Y_dpu[i],Y_host[i],Y_total[i], y_expected[rep][i]);
        //         }
        // }
        
        stop(&timer, 7);

        int s1=10;
        T verif=0;
        start(&timer, 8, rep);
        int powers = 1;
		for (unsigned int i = 0; i < m_size; i++){
            powers *= s1;
            verif+= (Y_total[i]) * powers;
   	     }
        stop(&timer, 8);
        //send result to DPU
        i = 0;
        mode = 1;
        start(&timer, 2, rep);
        DPU_FOREACH(dpu_set, dpu, i) {
        // Copy input arguments to DPU
            DPU_ASSERT(dpu_prepare_xfer(dpu, &mode));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "mode", 0, \
        sizeof(uint32_t), DPU_XFER_DEFAULT));
        i = 0;
        //end mode
        DPU_FOREACH(dpu_set, dpu, i) {
        // Copy input arguments to DPU
            DPU_ASSERT(dpu_prepare_xfer(dpu, Y_total + dpu_info[i].prev_rows_dpu));
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "total_prod", 0,max_rows_per_dpu  * sizeof(T), \
            DPU_XFER_DEFAULT));

        stop(&timer, 2);
        // Run DPU kernel
        start(&timer, 3, rep); 
        #if ENERGY
        DPU_ASSERT(dpu_probe_start(&probe)); 
        #endif
        // Launch kernel 
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS)); 
        stop(&timer, 3); 

        #if ENERGY
        DPU_ASSERT(dpu_probe_stop(&probe));
        #endif


#if PRINT
        {
            if (rep%200 == 0) {
                unsigned int each_dpu = 0;
                printf("Display DPU Logs\n");
                DPU_FOREACH (dpu_set, dpu) {
                    printf("DPU#%d:\n", each_dpu);
                    DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
                    each_dpu++;
                }
            }
        }
#endif
        //kernel on cpu
        T* gradient_cpu = calloc(n_size, sizeof(T));
        AES_init_ctx(&ctx, key);

        uint8_t* counter2 = malloc(((max_rows_per_dpu * nr_of_dpus * n_size_pad)/PART2 )* sizeof(uint8_t));
        // uint8_t* counter2 = malloc((max_rows_per_dpu * nr_of_dpus * n_size_pad)* sizeof(uint8_t));

        start(&timer, 9, rep);

        //  #pragma omp parallel for shared(Y, Y_total, gradient_cpu) private(counter2)
        for (int s = 0; s < PART2; s++) {
            // #pragma omp parallel for
            for (uint32_t i = 0; i < m_size/ PART2 ; i++) { //  
                int offset = s * (m_size / PART2) + i;
                for (unsigned int k = 0; k < n_size; k++) {
                    counter2[(i * n_size) + k] = (uint8_t)((((offset * n_size) + k)*sizeof(T)));//(uint8_t)(bufferX + (k * sizeof(T)));//[s*(max_rows_per_dpu * nr_of_dpus * n_size_pad/PART)+(i*(n_size_pad)+k)]);
                }
            }

            AES_ECB_encrypt(&ctx, counter2);

            for (uint32_t i = 0; i < m_size/ PART2; i++) {//  
                int offset = s * (m_size / PART2) + i;
                for (unsigned int k = 0; k < n_size; k++) {
                    gradient_cpu[k] -= counter2[(i * n_size) + k] * (Y[offset] - (Y_total[offset] >>
                            SHIFT_AMOUNT)); // y with offset
                }
            }
        }
        for (unsigned int k = 0; k < n_size; k++) {
            gradient_cpu[k]= gradient_cpu[k] >> (SHIFT_AMOUNT + OVERFLOW_SHIFT);
        }
                

        stop(&timer, 9);

        // Retrive result
        start(&timer, 4, rep); // DPU-CPU time 
        DPU_FOREACH(dpu_set, dpu, i) {
            DPU_ASSERT(dpu_prepare_xfer(dpu, gradient_dpu_tmp + i * n_size_pad)); 
        }
        DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0, n_size_pad * sizeof(T), \
            DPU_XFER_DEFAULT)); 
        stop(&timer, 4); 

        start(&timer, 5, rep); // CPU reduction time 
        // Compute gradient
        T* gradient_dpu = calloc(n_size, sizeof(T)); 
        T* total_gradient = calloc(n_size, sizeof(T)); 
        i = 0; 
        DPU_FOREACH(dpu_set, dpu, i) {
            for (uint32_t x = 0; x < n_size; ++x) {
                #ifdef FLOAT 
                gradient_dpu[x] += gradient_dpu_tmp[i*n_size_pad + x] / (int) m_size; 
                #else 
                gradient_dpu[x] += gradient_dpu_tmp[i*n_size_pad + x] >> OFFSET; 
                #endif 
            }
        } 
        //merge 
        // start4 = clock();
        start(&timer, 10, rep);
        for (uint32_t x = 0; x < n_size; ++x) {
            total_gradient[x]= gradient_dpu[x] + gradient_cpu[x];
            
        }
        // if(rep==1 || rep == 0){
        //         for(int i=0; i<10; i++){
        //             printf("gd_d: %d, gd_c: %d, gd_t: %d, gd_expexted: %d, y_real:%d \n", gradient_dpu[i],gradient_cpu[i],total_gradient[i], gd_expected[rep][i], bufferY[i]);
        //         }
        //     }
        
        stop(&timer,10);
        // int s2=10;
        T verifi=0;
        start(&timer, 11, rep);
		for (unsigned int i = 0; i < n_size; i++){
       			 verifi += (total_gradient[i]) * pow(s1,((n_size-i)));
        
   	     }
        stop(&timer, 11);

        // Update weight 
        for (uint32_t m = 0; m < n_size; ++m) { 
            #ifdef FLOAT 
            // float 
            W_dpu[m] = W_dpu[m] - (total_gradient[m]*learning_rate); 
            #else 
            // int 
            W_dpu_fp[m] = W_dpu_fp[m] - (total_gradient[m]*learning_rate) / (m_size >> OVERFLOW_SHIFT); 
            #endif 
        }
        // printf("iter: %d, gradient_dpu: %d, W_dpu_fp: %d\n", rep, gradient_dpu[0], W_dpu_fp[0]); 
        free(gradient_dpu); 
        free(gradient_cpu); 
        free(total_gradient); 
        stop(&timer, 5); // CPU reduction time 

        if (rep % 100 == 0)
            printf("DPU iter %d...\n", rep); 
    } // iter end 

    // Print trained weight at host 
    #ifdef INT32 
    float* W_host_float = (float*) malloc(n_size*sizeof(float));
    float* W_dpu_float  = (float*) malloc(n_size*sizeof(float));
    #endif
    printf("Trained weight at host: ");
    for (uint32_t x = 0; x < n_size; ++x) {
        #ifdef FLOAT
        printf("%.2f, ", (float) bufferW_host[x]); 
        #else
        W_host_float[x] = ((float) bufferW_host[x] / (SHIFT_MASK + 1)); 
        printf("%.2f; ", W_host_float[x]); 
        #endif
    }
    printf("\n"); 

    // Print DPU trained result 
    printf("Trained weight at DPU: ");
    for (uint32_t m = 0; m < n_size; ++m) {
        #ifdef FLOAT
        printf("%.2f, ", (float) W_dpu[m]); 
        #else
        // W_dpu_float[m] = (float) (W_dpu_fp[m]>>SHIFT_AMOUNT) + \
        //                     ((float)(W_dpu_fp[m]&SHIFT_MASK)/(1<<SHIFT_AMOUNT)); 
        W_dpu_float[m] = (float) W_dpu_fp[m] / (SHIFT_MASK + 1); 
        printf("%.2f, ", W_dpu_float[m]); 
        #endif
    }
    printf("\n"); 

    # ifdef FLOAT
    compute_mae(bufferX, bufferY, bufferW_host, m_size, n_size, "host"); 
    compute_mae(bufferX, bufferY, W_dpu, m_size, n_size, "DPUs"); 
    # else
    compute_mae(X_float, Y_float, W_host_float, m_size, n_size, "host"); 
    compute_mae(X_float, Y_float, W_dpu_float, m_size, n_size, "DPUs"); 
    free(X_float); 
    free(Y_float); 
    # endif

    #ifdef INT32 
    free(W_host_float);
    free(W_dpu_float); 
    #endif

    // Print timing results
    printf("CPU ");
    print(&timer, 0, 1);
    printf("init C-D ");
    print(&timer, 1, 1);
    printf("syn C-D ");
    print(&timer, 2, 1); 
    printf("DPU kernel ");
    print(&timer, 3, 1);
    printf("D-C ");
    print(&timer, 4, 1);
    printf("CPU reduction ");
    print(&timer, 5, 1);
    printf("CPU product ");
    print(&timer, 6, 1);
    printf("merge 1 ");
    print(&timer, 7, 1);
    printf("verif 1 ");
    print(&timer, 8, 1);
    printf("CPU gradient ");
    print(&timer, 9, 1);
    printf("merge 2 ");
    print(&timer, 10, 1);
    printf("verif 2 ");
    print(&timer, 11, 1);
    // printf("\n\nCPU PROD: %f\n", cpu_run);
    // printf("CPU gradient: %f\n", cpu_run2);
    // printf("merge1: %f\n", merge);
    // printf("merge2: %f\n", merge2);
    // printf("verif1: %f\n", verif1);
    // printf("verif2: %f\n", verif2);
    // printf("cpu: %f\n", cpu);

// #if ENERGY
//     double energy;
//     DPU_ASSERT(dpu_probe_get(&probe, DPU_ENERGY, DPU_AVERAGE, &energy));
//     printf("DPU Energy (J): %f\t", energy);
// #endif	

    // Check output
    bool status = true; 
    for (uint32_t each_attr = 0; each_attr < n_size; ++each_attr) {
        #ifdef FLOAT 
        if ((bufferW_host[each_attr] - W_dpu[each_attr] > 0.01) || 
            (bufferW_host[each_attr] - W_dpu[each_attr] < -0.01)) 
        {
            status = false; 
            # if PRINT
            printf("host: %.2f, dpu: %.2f\n", (float) bufferW_host[each_attr], (float) W_dpu[each_attr]); 
            #endif
        }
        #else
        if ((bufferW_host[each_attr]/(SHIFT_MASK+1) - W_dpu_fp[each_attr]/(SHIFT_MASK+1) > 0.01) || 
            (bufferW_host[each_attr]/(SHIFT_MASK+1) - W_dpu_fp[each_attr]/(SHIFT_MASK+1) < -0.01)) 
        {
            status = false; 
            // # if PRINT
            // printf("host: %.2f, dpu: %.2f\n", (float) bufferW_host[each_attr], (float) W_dpu[each_attr]); 
            // #endif
        }
        #endif
    }

    if (status) {
        printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
    } else {
        printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
    }

    // Deallocation
    free(input_args); 
    free(X_C);
    free(X_D);
    free(Y);
    free(W);
    free(W_dpu); 
    free(W_dpu_fp); 
    free(gradient_dpu_tmp); 
    DPU_ASSERT(dpu_free(dpu_set));
	
    return status ? 0 : -1;
}
