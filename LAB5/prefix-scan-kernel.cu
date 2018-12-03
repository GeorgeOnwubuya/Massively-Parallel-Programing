/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#define BLOCK_SIZE 512

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif



// Define your kernels in this file you may use more than one kernel if you
// need to

// INSERT KERNEL(S) HERE

__global__ void preScanKernel(float *out, float *in, unsigned size, float *sum){
    // INSERT CODE HERE
	__shared__ float a_s[(2 * BLOCK_SIZE) + CONFLICT_FREE_OFFSET(2 * BLOCK_SIZE)];
	int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	int thid = threadIdx.x;
	thid += CONFLICT_FREE_OFFSET(thid);
	int thid_BS = threadIdx.x + BLOCK_SIZE;
	thid_BS += CONFLICT_FREE_OFFSET(thid_BS);

	a_s[thid]    = ((idx              < size)? in[idx]:            0.0f);
        a_s[thid_BS] = ((idx + BLOCK_SIZE < size)? in[idx+BLOCK_SIZE]: 0.0f);


	unsigned int ai, bi;
	unsigned int numThreads, stride;

	for(numThreads = BLOCK_SIZE, stride = 1; numThreads > 0; numThreads >>= 1, stride <<= 1){

		ai = (2 * threadIdx.x * stride + stride - 1);
		bi = (2 * threadIdx.x * stride + 2 * stride - 1);

		ai += CONFLICT_FREE_OFFSET(ai);
		bi += CONFLICT_FREE_OFFSET(bi);

	__syncthreads();

		if(threadIdx.x < numThreads)
			a_s[bi] += a_s[ai];
	}

	if(threadIdx.x == 0){
		int last_elem = 2 * BLOCK_SIZE - 1;
		last_elem += CONFLICT_FREE_OFFSET(last_elem);
		if(sum != NULL){
			sum[blockIdx.x] = a_s[last_elem];
		}
		a_s[last_elem] = 0;
	}


	for(numThreads = 1, stride = BLOCK_SIZE; numThreads <= BLOCK_SIZE; numThreads <<= 1, stride >>= 1){

		ai = (2 * threadIdx.x * stride + stride - 1);
		bi = (2 * threadIdx.x * stride + 2 * stride - 1);

		ai += CONFLICT_FREE_OFFSET(ai);
		bi += CONFLICT_FREE_OFFSET(bi);

		__syncthreads();

		if(threadIdx.x < numThreads){
		float temp = a_s[bi];
		a_s[bi] += a_s[ai];
		a_s[ai] = temp;
		}
		__syncthreads();
	}
	if(idx < size)
	out[idx] = a_s[thid];

	if(idx + BLOCK_SIZE < size)
	out[idx + BLOCK_SIZE] = a_s[thid_BS];

}


__global__ void addKernel(float *out, float *sum, unsigned size)
{
    // INSERT CODE HERE
	int idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

	if(idx < size)
        out[idx] += sum[blockIdx.x];

        if(idx + BLOCK_SIZE < size)
        out[idx + BLOCK_SIZE] += sum[blockIdx.x];

}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void preScan(float *out, float *in, unsigned in_size)
{
	float *sum;
	unsigned num_blocks;
	cudaError_t cuda_ret;
	dim3 dim_grid, dim_block;

	num_blocks = in_size/(BLOCK_SIZE*2);
	if(in_size%(BLOCK_SIZE*2) !=0) num_blocks++;

	dim_block.x = BLOCK_SIZE; dim_block.y = 1; dim_block.z = 1;
	dim_grid.x = num_blocks; dim_grid.y = 1; dim_grid.z = 1;

	if(num_blocks > 1) {
		cuda_ret = cudaMalloc((void**)&sum, num_blocks*sizeof(float));
		if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

		preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, sum);
		preScan(sum, sum, num_blocks);
		addKernel<<<dim_grid, dim_block>>>(out, sum, in_size);

		cudaFree(sum);
	}
	else
		preScanKernel<<<dim_grid, dim_block>>>(out, in, in_size, NULL);
}
