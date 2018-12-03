/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SZ 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    unsigned int TiRow = threadIdx.y;
    unsigned int TiCol = threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[TILE_SZ][TILE_SZ];
    __shared__ float Bs[TILE_SZ][TILE_SZ];

    float sum = 0;


    for(unsigned int TiNum = 0; TiNum < (k-1)/TILE_SZ+1; TiNum++){
  	if((row < m) && (TiNum * TILE_SZ + TiCol) < k)
    		As[TiRow][TiCol]= A[row * k + TiNum * TILE_SZ + TiCol];
  	else
    		As[TiRow][TiCol] = 0;

 	if((TiNum * TILE_SZ + TiRow) < k && col < n)
    		Bs[TiRow][TiCol] = B[(TiNum * TILE_SZ + TiRow) * n + col];
	else
    		Bs[TiRow][TiCol] = 0;
  	__syncthreads();

        //Calculate inner product for the tile
        //Checking for matrix size to lower power and practice green computing
        if(row < m && col < n)
		for(unsigned int TiElem = 0; TiElem < TILE_SZ; TiElem++)
        		sum = sum + As[TiRow][TiElem]*Bs[TiElem][TiCol];
        __syncthreads();

        } 
    
        //Prevent writing of output to an undefined block
        if (row < m && col < n)
        	C[row * n + col] = sum;
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------
  
   //INSERT CODE HERE
    const unsigned int BLOCK_SIZE = TILE_SZ; //use 16 x 16 thread blocks
   
    dim3 block(BLOCK_SIZE, BLOCK_SIZE ,1);
    dim3 grid((n + BLOCK_SIZE - 1)/BLOCK_SIZE, (m + BLOCK_SIZE -1)/BLOCK_SIZE, 1);  
  

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE

    mysgemm<<< grid, block>>>(m, n, k, A, B, C);

}
