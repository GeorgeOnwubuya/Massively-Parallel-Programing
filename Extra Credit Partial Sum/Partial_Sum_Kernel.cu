
__global__ void vecSum_final_int(int * array)
{


  for(unsigned int offset = blockDim.x; offset  > 0; offset = offset >> 1){
      __syncthreads();

      if (threadIdx.x < offset)
          array[threadIdx.x] += array[threadIdx.x + offset];
  }
}

__global__ void vecSum_final_int1(int * array)
{
  const int tidx = threadIdx.x << 1;

  for (unsigned int stride = 1; stride <= blockDim.x; stride = stride << 1 ){

      __syncthreads();

      if(tidx % stride == 0)
         array[tidx] += array[tidx + stride];

 }
}
