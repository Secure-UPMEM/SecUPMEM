#include <stdlib.h>
#include <stdio.h>
#include "../../support/timer.h"
//#include "../../support/params.h"
#include "../../support/aes.h"
#include "gemv_utils.h"
#define T uint32_t
//int batch =600;
/*typedef struct Params {
    unsigned int  m;
    unsigned int  n;
}Params;
struct Params input_params(int argc, char **argv) {
    struct Params p;
    p.m        = 4000;
    p.n        = 4000;

    int opt;
    while((opt = getopt(argc, argv, "hm:n:w:e:")) >= 0) {
        switch(opt) {
            case 'h':
                usage();
                exit(0);
                break;
            case 'm': p.m        = atoi(optarg); break;
            case 'n': p.n        = atoi(optarg); break;
            default:
                      fprintf(stderr, "\nUnrecognized option!\n");
                      usage();
                      exit(0);
        }
    }
    //assert(NR_DPUS > 0 && "Invalid # of dpus!");

    return p;
}*/

int main(int argc, char *argv[])
{
	//struct Params p = input_params(argc, argv);
  const size_t m_size = 4000;//p.m;
  const size_t n_size = 4000;//p.n;
printf("start\n");
  T **A, *b, *x;
	int batch=600;
  b = (T*)  malloc(sizeof(T)*m_size);
  x = (T*) malloc(sizeof(T)*n_size);
printf("ok\n");
  allocate_dense(m_size, n_size, &A);

  make_hilbert_mat(m_size,n_size, &A);
	printf("allocation\n");
#pragma omp parallel
    {
#pragma omp for
    for (size_t i = 0; i < n_size; i++) {
      x[i] = (T) i+1 ;
    }

#pragma omp for
    for (size_t i = 0; i < m_size; i++) {
      b[i] = (T) 0.0;
    }
    }	
	/*********************************SecNDP ***************************************/
	uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	struct AES_ctx ctx;
	float sec=0;
	uint8_t s1[]={ 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};
	uint8_t* first = (T*) malloc(sizeof(T)*m_size*n_size);
    	T* ciphertext = (T*) malloc(sizeof(T)*m_size*n_size);
    	uint32_t* temp = (T*) malloc(sizeof(T)*m_size*n_size);
    	//T* C_host = (T*) malloc(sizeof(T)*m_size);
    	printf("start\n");
    	//startTimer(&timer1);
    	int count=0;
    	for(int i=0;i< m_size; i++){
    		for(int j=0;j< n_size; j++){

		 first[count] = (uint8_t)(&A[i][j]);
		 count++;
		 }
        //printf("%d  \n", ciphertext[i]);
    	}
    	AES_init_ctx(&ctx, key);
    	AES_ECB_encrypt(&ctx, first);
	//for(int b=0; b< batch; b++){
    	for(int i=0;i< m_size*n_size; i++){

        	ciphertext[i] = (uint32_t)first[i];
		temp[i]= (A[i%m_size][i%n_size]) - ciphertext[i];	
		//printf("a: %d , temp: %d, cipher: %d\n", A[i], temp[i], ciphertext[i]);
     	}
     	count =0;
     	
     	for(int i=0;i< m_size; i++){
    		for(int j=0;j< n_size; j++){

		 A[i][j] = temp[count];
		 count++;
		 }
        //printf("%d  \n", ciphertext[i]);
    	}
     	//}
	printf("sec done\n");
     	//stopTimer(&timer1);
    	//sec += getElapsedTime(timer1);
	/*********************************SecNDP****************************************/
	
	
  Timer timer;
  start(&timer, 0, 0);
	count =0;
	for(int i=0;i< m_size; i++){
    		for(int j=0;j< n_size; j++){

		 first[count] = (uint8_t)(&A[i][j]);
		 count++;
		 }
        //printf("%d  \n", ciphertext[i]);
    	}
    	AES_init_ctx(&ctx, key);
    	AES_ECB_encrypt(&ctx, first);
    	count =0;
	for(int s=0; s< batch; s++){
    	for(int i=0;i< m_size; i++){
    		for(int j=0;j< n_size; j++){

		 A[i][j] = A[i][j] + first[count];
		 count++;
		 }
        //printf("%d  \n", ciphertext[i]);
    	}}
    stop(&timer, 0);


    printf("Decryption ");
    print(&timer, 0, 1);
    printf("\n");
    
    start(&timer, 0, 0);
    for(int s=0;s<batch;s++){
   gemv(A, x, m_size, n_size, &b);
   }
   stop(&timer, 0);


    printf("Kernel ");
    print(&timer, 0, 1);
    printf("\n");

#if 0
  print_vec(x, rows);
  print_mat(A, rows, cols);
  print_vec(b, rows);
#endif

  //printf("sum(x) = %f, sum(Ax) = %f\n", sum_vec(x,cols), sum_vec(b,rows));
  return 0;
}

void gemv(T** A, T* x, size_t rows, size_t cols, T** b) {
#pragma omp parallel for
  for (size_t i = 0; i < rows; i ++ )
  for (size_t j = 0; j < cols; j ++ ) {
    (*b)[i] = (*b)[i] + A[i][j]*x[j];
  }
}

void make_hilbert_mat(size_t rows, size_t cols, T*** A) {
#pragma omp parallel for
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      (*A)[i][j] = 1.0/( (T) i + (T) j + 1.0);
    }
  }
}

T sum_vec(T* vec, size_t rows) {
  T sum = 0.0;
#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < rows; i++) sum = sum + vec[i];
  return sum;
}
