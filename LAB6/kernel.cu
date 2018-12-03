/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

// Define your kernels in this file you may use more than one kernel if you
// need to
__global__ void histogram_kernel(unsigned int* input, unsigned int* bins, 
    unsigned int num_elements, unsigned int num_bins){

	extern __shared__ unsigned int bins_s[];
	
	//Shared Memory
	int thid = threadIdx.x;
	while(thid < num_bins){

		bins_s[thid] = 0u;
		thid += blockDim.x;
	}
	__syncthreads();


	//Histogram calculation
	unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
	
	while(element < num_elements){

		atomicAdd(&(bins_s[input[element]]), 1);
		element += blockDim.x * gridDim.x;
	}
	__syncthreads();

	//Global Memory
	thid = threadIdx.x;
	while(thid < num_bins){

		atomicAdd(&(bins[thid]), bins_s[thid]);
		thid += blockDim.x;
	} 
}
 

__global__ void histogram_kernel_optimized(unsigned int* input, unsigned int* bins,
    unsigned int num_elements, unsigned int num_bins) {

      	// INSERT CODE HERE 
	extern __shared__ unsigned int bins_s[];
	
	//Shared memory	
	int thid = threadIdx.x;
	while ( thid < num_bins){

		bins_s[thid] = 0u;
		thid += blockDim.x; 
	}
	__syncthreads();	
 
	//Histogram calculation
	unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int accumulator = 0;
	unsigned int prev_index = 0;
	 
	while(element < num_elements){
		
		unsigned int curr_index = input[element];
		
		if(curr_index != prev_index){
				
			atomicAdd(&(bins_s[prev_index]), accumulator);
			accumulator = 1;
			prev_index = curr_index;
		
		}	
			
		else{
			accumulator++;	
		}
		element += blockDim.x * gridDim.x;
	}
	if(accumulator > 0){
		atomicAdd(&(bins_s[prev_index]), accumulator);
	}
	__syncthreads();

	//Global memory
	thid = threadIdx.x;
	while(thid < num_bins){
	
		atomicAdd(&(bins[thid]), bins_s[thid]);
		thid += blockDim.x;
	}

}

__global__ void convert_kernel(unsigned int *bins32, uint8_t *bins8,
    unsigned int num_bins) {

      // INSERT CODE HERE
	int thid = blockIdx.x * blockDim.x + threadIdx.x;

	while (thid < num_bins){
	
		//Use local  register value (avoids copying from global twice)		
		unsigned int reg_bin = bins32[thid];
		
		if(reg_bin > 255){
			bins8[thid] = 255u;
		}

		else{
			bins8[thid] = (uint8_t) reg_bin;
		}
		thid += blockDim.x * gridDim.x;
	}

}

/******************************************************************************
Setup and invoke your kernel(s) in this function. You may also allocate more
GPU memory if you need to
*******************************************************************************/
void histogram(unsigned int* input, uint8_t* bins, unsigned int num_elements,
        unsigned int num_bins) {

    // Create 32 bit bins
    unsigned int *bins32;
    cudaMalloc((void**)&bins32, num_bins * sizeof(unsigned int));
    cudaMemset(bins32, 0, num_bins * sizeof(unsigned int));

    // Launch histogram kernel using 32-bit bins
    dim3 dim_grid, dim_block;
    dim_block.x = 512; dim_block.y = dim_block.z = 1;
    dim_grid.x = 30; dim_grid.y = dim_grid.z = 1;

    //Comment out the kernel not used
    //histogram_kernel<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>
       // (input, bins32, num_elements, num_bins);
   histogram_kernel_optimized<<<dim_grid, dim_block, num_bins*sizeof(unsigned int)>>>
	(input, bins32, num_elements, num_bins);

    // Convert 32-bit bins into 8-bit bins
    dim_block.x = 512;
    dim_grid.x = (num_bins - 1)/dim_block.x + 1;
    convert_kernel<<<dim_grid, dim_block>>>(bins32, bins, num_bins);

    // Free allocated device memory
    cudaFree(bins32);

}
