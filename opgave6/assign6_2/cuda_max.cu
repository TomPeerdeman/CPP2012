#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <iostream>

#include "timer.h"

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
__global__ void maxKernel(int length, float *list, float *max) {
  float temp;
  
  // Global means the position in the list
  int globalIdx1;
  int globalIdx2;
  // Local means position within the block
  int localIdx2;
  
  int halfPoint;

  // calculate number of threads needed in the first iteration
  int nTotalThreads = NearestPowerOf2(blockDim.x);

  while(nTotalThreads > 1){
    // we only need the first half of the array, we compare with the other half.
    halfPoint = nTotalThreads / 2;
	
	globalIdx1 = threadIdx.x + blockIdx.x * blockDim.x;
	
    // see if i am in the first half
    if (threadIdx.x < halfPoint){
      // i have to compare with the second half of the array,
      // my id + half the length of the remaining list
      localIdx2 = threadIdx.x + halfPoint;
	  globalIdx2 = localIdx2 + blockIdx.x * blockDim.x;

      // only work in the same block
      if (localIdx2 < blockDim.x && globalIdx2 < length){

        temp = list[globalIdx2];
        // the highest value goes to the front part of the remaining list.
        if (temp > list[globalIdx1])
           list[globalIdx1] = temp;
      }
    }
    __syncthreads();

	// The max of this block is placed the local first place of this block
	if(threadIdx.x == 0){
		max[blockIdx.x] = list[globalIdx1];
	}
   
    // next iteration will be done with half the length we had before.
    nTotalThreads = halfPoint;
  }
  
  //
}

float computeMaxCuda(int length, int block_size, int tpb, float* list){
  float* d_list = NULL;
  float* d_max = NULL;
  timer maxTimer("Max timer");

  // Alloc space on the device.
  checkCudaCall(cudaMalloc((void **) &d_list, length * sizeof(float)));
  checkCudaCall(cudaMalloc((void **) &d_max, block_size * sizeof(float)));
  // copy memory to device for parallelism
  checkCudaCall(cudaMemcpy(d_list, list, length*sizeof(float), cudaMemcpyHostToDevice));

  // start timer (only time the calculation of the max value,
  // including the list will make the time increase.
  maxTimer.start();
  
  // preform CUDA parallelism
  maxKernel<<<block_size,tpb>>>(length, d_list, d_max);
  
  // stop time
  maxTimer.stop();
    
  // copy memory back from device
  checkCudaCall(cudaMemcpy(list, d_list, sizeof(float)*length, cudaMemcpyDeviceToHost));

  // Free device mem.
  checkCudaCall(cudaFree(d_list));
  checkCudaCall(cudaFree(d_max));
  
  // show time needed for the calculation
  cout << maxTimer;
  
  // return value for comparison later
  return list[0];
}

