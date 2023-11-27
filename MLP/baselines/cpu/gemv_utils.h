#define T uint32_t
void allocate_dense(size_t rows,size_t  cols, T*** dense) {

  *dense = malloc(sizeof(T)*rows);
  **dense = malloc(sizeof(T)*rows*cols);

  for (size_t i=0; i < rows; i++ ) {
    (*dense)[i] = (*dense)[0] + i*cols;
  }

}

void print_mat(T** A, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      printf("%f ", A[i][j]);
    }
    printf("\n");
  }
}

void print_vec(T* b, size_t rows) {
  for (size_t i = 0; i < rows; i++) {
    printf("%f\n", b[i]);
  }
}

void gemv(T** A, T* x, size_t rows, size_t cols, T** b);
void make_hilbert_mat(size_t rows, size_t cols, T*** A);
T sum_vec(T* vec, size_t rows);
