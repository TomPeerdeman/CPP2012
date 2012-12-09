#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <iostream>

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}

__device__ int NearestPowerOf2(int n)
{
  if (!n) return n;  //(0 == 2^0)
 
  int x = 1;
  while(x < n)
    {
      x <<= 1;
    }
  return x;
}


// standard binary tree reduction cuda method
__global__ void maxKernel(const int block_size) {
  int  thread2;
  float temp;
  extern __shared__ float max[];
   
  int nTotalThreads = NearestPowerOf2(blockDim.x);	// Total number of threads, rounded up to the next power of two
   
  while(nTotalThreads > 1)
  {
    int halfPoint = (nTotalThreads >> 1);	// divide by two
    // only the first half of the threads will be active.
   
    if (threadIdx.x < halfPoint)
    {
     thread2 = threadIdx.x + halfPoint;
   
     // Skipping the fictious threads blockDim.x ... blockDim_2-1
     if (thread2 < blockDim.x)
       {
   
        temp = max[thread2];
        if (temp > max[threadIdx.x]) 
           max[threadIdx.x] = temp;
       }
    }
    __syncthreads();
   
    // Reducing the binary tree size by two:
    nTotalThreads = halfPoint;
  }
}

// TODO create working compute function that returns the max value of an array
float *computeMaxCuda(int length, int block_size, int tpb){
    
  float* d_list = NULL;
  float* d_max = NULL;
  float* maxVal = NULL;
  float list[length];
  int max_size = block_size
  srand(time(NULL));
  //TODO make this run in parallel
  //printfs, so test with small length!
  for(int i = 0; i< length; i++){
    list[i] = (float)rand()/((float)RAND_MAX/FLT_MAX);
    printf("%d). %lf\n",i,list[i]);
  }
  
	// Alloc space on the device.
	// Is this the right amount?
	checkCudaCall(cudaMalloc((void **) &d_list, length * sizeof(float)));
	checkCudaCall(cudaMalloc((void **) &d_max, sizeof(float)));
	
  maxKernel<<<block_size, tpb, max_size>>>(block_size);
  printf("max value?: %lf", max[0]);
  // copy resulting max back to main memory
  checkCudaCall(cudaMemcpy(maxVal, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	// Free device mem.
	checkCudaCall(cudaFree(d_list));
	checkCudaCall(cudaFree(d_max));
	
	return maxVal;
}
