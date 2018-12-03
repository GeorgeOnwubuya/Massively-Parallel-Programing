/********************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

__constant__ float M_c[FILTER_SIZE][FILTER_SIZE];

/*__device__ float getElement(Matrix *N, const int row, const int col)
{
	return N->elements[row*N->width+col];
}
*/
/*__device__ void retElem(Matrix *P, const int row, const int col, float value)
{
	P->elements[row*P->width+col] = value; 

	return;
}*/

__global__ void convolution(Matrix N, Matrix P)
{
	/********************************************************************
	Determine input and output indexes of each thread
	Load a tile of the input image to shared memory
	Apply the filter on the input image tile
	Write the compute values to the output image at the correct indexes
	********************************************************************/

        //INSERT KERNEL CODE HERE
	/*int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col_zeroIndex = col - FILTER_SIZE/2;
	int row_zeroIndex = row - FILTER_SIZE/2;
	float sum = 0;

	for(int j = 0; j < FILTER_SIZE; ++j){
		for(int k = 0; k < FILTER_SIZE; ++k){
			if((row_zeroIndex + j >= 0) && (row_zeroIndex + j < N.height) &&   	   		
		   	  (col_zeroIndex + k >= 0) &&  (col_zeroIndex+ k < N.width)){		
					//sum = M_c[j][k] * getElement(&N, row_zeroIndex + j, col_zeroIndex + k);
			
					sum += M_c[j][k] * N.elements[(row_zeroIndex + j)*N.width + col_zeroIndex +k];				
			}
		}
 	}
        if( row < P.height  && col < P.width)
		
		//retElem(&P, row, col, sum);

		P.elements[row * P.width + col] = sum;*/


	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;
	int rowZeroIndex = row - FILTER_SIZE/2;
	int colZeroIndex = col - FILTER_SIZE/2;


	__shared__ float N_ds[TILE_SIZE + FILTER_SIZE - 1][TILE_SIZE + FILTER_SIZE - 1];

	if((rowZeroIndex >= 0) && (rowZeroIndex < N.height) && (colZeroIndex >= 0) && (colZeroIndex < N.width)){
		
		N_ds[threadIdx.y][threadIdx.x] = N.elements[rowZeroIndex * N.width + colZeroIndex];
	}

	else{
		N_ds[threadIdx.y][threadIdx.x] = 0.0f;
	}

	__syncthreads();
	
	float sum = 0.0f;
	
	if(threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE){

		for(int dr = 0; dr < FILTER_SIZE; ++dr){

			for(int dc = 0; dc < FILTER_SIZE; ++dc){

				sum += M_c[dr][dc] * N_ds[threadIdx.y + dr][threadIdx.x + dc];
			}
		}

	if(row < P.height && col < P.width){
          
        	P.elements[row * P.width + col] = sum;	
	}		
	
	}
}


