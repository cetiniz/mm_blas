
#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include <stdio.h>
#include <stdlib.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 8
#endif

#define MU 4
#define NU 4
#define KU 4

#define min(a,b) (((a)<(b))?(a):(b))

 static double packed_A[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (16)));
 static double packed_B[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (16)));
 static double packed_C[BLOCK_SIZE*BLOCK_SIZE] __attribute__ ((aligned (16)));

 static void print4x4(double* mat, int lda) {
   for (int i=0;i<4;i++) {
    for (int j=0;j<4;j++) {
      printf("%f | ", mat[i*lda+j]);
   }
   printf("\n");
  }
  printf("\n");
 }

typedef union
{
   double d[4];
   __m256d v;
} d256_union;

 static void micro_nest(int lda, int M, int N, int K, double* restrict A, double* restrict B, double* restrict C) {
  d256_union row_1, row_2, row_3, row_4;
  row_1.v = _mm256_setzero_pd();
  row_2.v = _mm256_setzero_pd();
  row_3.v = _mm256_setzero_pd();
  row_4.v = _mm256_setzero_pd();

  for (int i = 0; i < 4; i++) {
    __m256d a_broadcast = _mm256_set1_pd(B[BLOCK_SIZE*0+i]);
    __m256d a_row = _mm256_set_pd(A[i*BLOCK_SIZE], A[i*BLOCK_SIZE+1], A[i*BLOCK_SIZE+2], A[i*BLOCK_SIZE+3]);
    row_1.v = _mm256_fmadd_pd(a_row, a_broadcast, row_1.v);
 
    __m256d b_broadcast = _mm256_set1_pd(B[BLOCK_SIZE*1+i]);
    __m256d b_row = _mm256_set_pd(A[i*BLOCK_SIZE], A[i*BLOCK_SIZE+1], A[i*BLOCK_SIZE+2], A[i*BLOCK_SIZE+3]);
    row_2.v = _mm256_fmadd_pd(b_row, b_broadcast, row_2.v);
 
    __m256d c_broadcast = _mm256_set1_pd(B[BLOCK_SIZE*2+i]);
    __m256d c_row = _mm256_set_pd(A[i*BLOCK_SIZE], A[i*BLOCK_SIZE+1], A[i*BLOCK_SIZE+2], A[i*BLOCK_SIZE+3]);
    row_3.v = _mm256_fmadd_pd(c_row, c_broadcast, row_3.v);
 
    __m256d d_broadcast = _mm256_set1_pd(B[BLOCK_SIZE*3+i]);
    __m256d d_row = _mm256_set_pd(A[i*BLOCK_SIZE], A[i*BLOCK_SIZE+1], A[i*BLOCK_SIZE+2], A[i*BLOCK_SIZE+3]);
    row_4.v = _mm256_fmadd_pd(d_row, d_broadcast, row_4.v);
   }

  //  C[0]     += row_1d[3]; C[1]         += row_1d[2]; C[2]         += row_1d[1]; C[3]         += row_1d[0];
  //  C[lda]   += row_2d[3]; C[lda + 1]   += row_2d[2]; C[lda + 2]   += row_2d[1]; C[lda + 3]   += row_2d[0];
  //  C[lda*2] += row_3d[3]; C[lda*2 + 1] += row_3d[2]; C[lda*2 + 2] += row_3d[1]; C[lda*2 + 3] += row_3d[0];
  //  C[lda*3] += row_4d[3]; C[lda*3 + 1] += row_4d[2]; C[lda*3 + 2] += row_4d[1]; C[lda*3 + 3] += row_4d[0];
  _mm256_store_pd(&C[0], row_1.v); _mm256_store_pd(&C[lda], row_2.v); _mm256_store_pd(&C[lda*2], row_3.v); _mm256_store_pd(&C[lda*3], row_4.v);
}

 static void printPA() {
   printf("PACKED_A\n");
   for (int i=0; i< BLOCK_SIZE; i++) {
     for (int j=0; j< BLOCK_SIZE; j++){
       printf("%10f | ", packed_A[i*BLOCK_SIZE+j]);
       if ((j+1) % 4 == 0 && j != 0) {
         printf("    ");
       }
     }
     printf("\n");
     if (i == 3) {
       printf("\n");
     }
   }
 }

 static void printPB() {
   printf("PACKED_B\n");
   for (int i=0; i< BLOCK_SIZE; i++) {
     for (int j=0; j< BLOCK_SIZE; j++){
       printf("%10f | ", packed_B[i*BLOCK_SIZE+j]);
       if ((j+1) % 4 == 0 && j != 0) {
         printf("    ");
       }
     }
     printf("\n");
     if (i == 3) {
       printf("\n");
     }
   }
 }

 static void do_block (int lda, int M, int N, int K, double* C) {
   for (int i = 0; i < M; i+=MU) {
   int reg_m = min(MU, M-i);
     for (int j = 0; j < N; j+=NU) {
       int reg_n = min(NU, N-j);
       for (int k = 0; k < K; k+=KU) {
         int reg_k = min(KU, K-k);
         micro_nest(lda, reg_m, reg_n, KU, packed_A+j*BLOCK_SIZE+k, packed_B+i*BLOCK_SIZE+j, C + k + i*lda);
       }
     }
   }
 }


 void pack_A(int lda, double* B, int num_rows, int num_cols) {
    int col_overflow = num_cols % BLOCK_SIZE == 0 ? BLOCK_SIZE : num_cols % BLOCK_SIZE;
    int row_overflow = num_rows % BLOCK_SIZE == 0 ? BLOCK_SIZE : num_rows % BLOCK_SIZE;
    int i, j;
 
   if (col_overflow == BLOCK_SIZE) {
     for (i = 0; i < BLOCK_SIZE; i++) {
       for (j = 0; j < BLOCK_SIZE; j++) {
         packed_A[j*BLOCK_SIZE+i] = B[i + j*lda];
       }
     }
   }

   if (col_overflow != BLOCK_SIZE || row_overflow != BLOCK_SIZE) {
     for (i = 0; i < row_overflow; i++) {
       for (j = 0; j < col_overflow; j++) {
         packed_A[i*BLOCK_SIZE+j] = B[j + i*lda];
       }
       for (j = col_overflow; j < BLOCK_SIZE; j++) {
         packed_A[i*BLOCK_SIZE + j] = 0.0;
       }
     }
     for (i =row_overflow; i < BLOCK_SIZE; i++) {
       for(j = 0; j < BLOCK_SIZE; j++) {
         packed_A[i*BLOCK_SIZE + j] = 0.0;
       }
     }
   }
 }

void pack_B(int lda, double* B, int num_rows, int num_cols) {
    int col_overflow = num_cols % BLOCK_SIZE == 0 ? BLOCK_SIZE : num_cols % BLOCK_SIZE;
    int row_overflow = num_rows % BLOCK_SIZE == 0 ? BLOCK_SIZE : num_rows % BLOCK_SIZE;
    int i, j;
 
   if (col_overflow == BLOCK_SIZE) {
     for (i = 0; i < BLOCK_SIZE; i++) {
       for (j = 0; j < BLOCK_SIZE; j++) {
         packed_B[j*BLOCK_SIZE+i] = B[i + j*lda];
       }
     }
   }

   if (col_overflow != BLOCK_SIZE || row_overflow != BLOCK_SIZE) {
     for (i = 0; i < row_overflow; i++) {
       for (j = 0; j < col_overflow; j++) {
         packed_B[i*BLOCK_SIZE+j] = B[j + i*lda];
       }
       for (j = col_overflow; j < BLOCK_SIZE; j++) {
         packed_B[i*BLOCK_SIZE + j] = 0.0;
       }
     }
     for (i =row_overflow; i < BLOCK_SIZE; i++) {
       for(j = 0; j < BLOCK_SIZE; j++) {
         packed_B[i*BLOCK_SIZE + j] = 0.0;
       }
     }
   }
 }

 void square_dgemm (int lda, double* A, double* B, double* C) {
   for (int i = 0; i < lda; i += BLOCK_SIZE) {
     int M = min (BLOCK_SIZE, lda-i); // Will control B ROW
     for (int j = 0; j < lda; j += BLOCK_SIZE) {
       int N = min (BLOCK_SIZE, lda-j);
       pack_B(lda, B + lda*i + j, M, N);
       for (int k = 0; k < lda; k += BLOCK_SIZE) {
         int K = min (BLOCK_SIZE, lda-k);
         pack_A(lda, A + lda*j + k, N, K);
         do_block(lda, M, N, K, C + k + i*lda);
       }
     }
   }
 }

int main(int argc, char const *argv[]) {
  const int MATRIX_SIZE = 1000;

  double counter = 1;
  double* matrix = (double*) malloc(MATRIX_SIZE*MATRIX_SIZE * sizeof(double));
  double* matrix_c = (double*) malloc(MATRIX_SIZE*MATRIX_SIZE * sizeof(double));
  for (int i = 0; i < MATRIX_SIZE; i++) {
    for (int j = 0; j < MATRIX_SIZE; j++) {
      matrix[(j*MATRIX_SIZE)+i] = rand() % 20; //counter+=1.0;
      matrix_c[(j*MATRIX_SIZE)+i] = 0;
    }
  }

  // for (int i = 0; i < MATRIX_SIZE; i++) {
  //   for (int j = 0; j < MATRIX_SIZE; j++) {
  //     printf("%f | ", matrix[i*MATRIX_SIZE+j]);
  //     if (j == MATRIX_SIZE - 1) {
  //       printf("\n");
  //     }
  //   }
  // }
  // printf("\n");
  // for (int i = 0; i < MATRIX_SIZE; i+=BLOCK_SIZE) {
  //   for (int j = 0; j < MATRIX_SIZE; j+=BLOCK_SIZE) {
  //     int num_rows = MATRIX_SIZE - i < BLOCK_SIZE ? MATRIX_SIZE -i : BLOCK_SIZE;
  //     int num_cols = MATRIX_SIZE - j < BLOCK_SIZE ? MATRIX_SIZE -j : BLOCK_SIZE;
  //     pack_A(MATRIX_SIZE, matrix+j+i*MATRIX_SIZE, num_rows, num_cols);
  //     printf("MATRIX_A\n");
  //     printf("PACKING ROWS %d-%d   |   PACKING COLS %d-%d\n", i, i+BLOCK_SIZE, j, j+BLOCK_SIZE);

  //     for (int z = 0; z < BLOCK_SIZE; z++) {
  //       for (int y = 0; y < BLOCK_SIZE; y++) {
  //         printf("%f | ", packed_A[z*BLOCK_SIZE + y]);
  //         if (y == BLOCK_SIZE - 1) {
  //           printf("\n");
  //         }
  //       }
  //     }
  //   }
  // }
  // for (int i = 0; i < MATRIX_SIZE; i+=BLOCK_SIZE) {
  //   for (int j = 0; j < MATRIX_SIZE; j+=BLOCK_SIZE) {
  //     int num_rows = MATRIX_SIZE - i < BLOCK_SIZE ? MATRIX_SIZE -i : BLOCK_SIZE;
  //     int num_cols = MATRIX_SIZE - j < BLOCK_SIZE ? MATRIX_SIZE -j : BLOCK_SIZE;
  //     pack_B(MATRIX_SIZE, matrix+j+i*MATRIX_SIZE, num_rows, num_cols);
  //     printf("MATRIX_B\n");
  //     printf("PACKING ROWS %d-%d   |   PACKING COLS %d-%d\n", i, i+BLOCK_SIZE, j, j+BLOCK_SIZE);

  //     for (int z = 0; z < BLOCK_SIZE; z++) {
  //       for (int y = 0; y < BLOCK_SIZE; y++) {
  //         printf("%f | ", packed_B[z*BLOCK_SIZE + y]);
  //         if (y == BLOCK_SIZE - 1) {
  //           printf("\n");
  //         }
  //       }
  //     }
  //   }
  // }
  square_dgemm(MATRIX_SIZE, matrix, matrix, matrix_c);
  // for (int i = 0; i < MATRIX_SIZE; i++) {
  //   for (int j = 0; j < MATRIX_SIZE; j++) {
  //     printf("%10.2f | ", matrix_c[i*MATRIX_SIZE+j]);
  //     if (j == MATRIX_SIZE - 1) {
  //       printf("\n");
  //     }
  //   }
  // }
  free(matrix);
  free(matrix_c);
  return 0;
}
