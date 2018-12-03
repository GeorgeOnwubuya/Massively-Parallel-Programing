#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Partial_Sum_Kernel.cu"

#define BLOCK_SIZE  32;
#define SAMPLE_SIZE 32

void FATAL (const char * s )
{
	puts(s);
	exit(1);
}

int main(int argc, char**argv) {
    
    unsigned int data_size;
    cudaError_t cuda_ret;

    
    if(argc == 1) {
        data_size= 64;
    } else if(argc == 2) {
        data_size= atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }

//Device data
int * array_dev;
int array_size = data_size;

//Host data
int * array_host = (int *) malloc (sizeof(int)*array_size);
for(int i = 0; i < data_size; ++i)
	array_host[i] = i + 1;

for (int i = data_size; i < array_size; ++i)
	array_host[i] = 0;

int expected_sum = data_size * (array_host[0] + array_host[data_size - 1]) / 2;

//Allocating & copying device memory
cuda_ret = cudaMalloc((void**)&array_dev, array_size*sizeof(int));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
cuda_ret = cudaMemcpy(array_dev, array_host, array_size*sizeof(int), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to device");

cudaDeviceSynchronize();

//Invoke Kernel}
vecSum_final_int1<<<dim3(1, 1, 1), dim3(SAMPLE_SIZE, 1, 1)>>>(array_dev);
vecSum_final_int<<<dim3(1, 1, 1), dim3(SAMPLE_SIZE, 1, 1)>>>(array_dev);

//Copying to host memory
int *result = (int *) malloc(sizeof(int)*array_size);
cuda_ret = cudaMemcpy(result, array_dev, sizeof(int)*array_size, cudaMemcpyDeviceToHost);
if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

cudaDeviceSynchronize();

printf("Array size = %d\n", array_size);
printf("Expected result = %d\n", expected_sum);
printf("Calculated result = %d\n", result[0]);

for (int i = 0; i < data_size; ++i){
	printf("[%2d] : %5d, %5d\n", i, array_host[i], result[i]);
}


fflush(stdout);

free(array_host);
cudaFree(array_dev);

return 0;

};
