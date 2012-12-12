#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
__global__ void maxKernel(float* maxList) {
  int  thread2;
  float temp;

  // calculate number of threads needed in the first iteration
  int nTotalThreads = NearestPowerOf2(blockDim.x);

  while(nTotalThreads > 1)
  {
    // we only need the first half of the array, we compare with the other half.
    int halfPoint = nTotalThreads / 2;

    // see if i am in the first half
    if (threadIdx.x < halfPoint){
      // i have to compare with the second half of the array,
      // my id + half the length of the remaining list
      thread2 = threadIdx.x + halfPoint;

      // only work in the same block?
      if (thread2 < blockDim.x){

        temp = maxList[thread2];
        // the highest value goes to the front part of the remaining list.
        if (temp > maxList[threadIdx.x])
           maxList[threadIdx.x] = temp;
      }
    }
    __syncthreads();

   
    // next iteration will be done with half the length we had before.
    nTotalThreads = halfPoint;
  }
}

void computeMaxCuda(int length, int block_size, int tpb){

  float* d_list = NULL;
  float* d_max = NULL;
  float list[length];
  timer maxTimer("Max timer");
  srand(time(NULL));

  // make a list of floats
  for(int i = 0; i< length; i++){
    list[i] = (float)rand()/((float)RAND_MAX/FLT_MAX);
  }
  
  // Alloc space on the device.
  checkCudaCall(cudaMalloc((void **) &d_list, length * sizeof(float)));
  checkCudaCall(cudaMalloc((void **) &d_max, sizeof(float)));
  // copy memory to device for parallelism
  checkCudaCall(cudaMemcpy(d_list, list, length*sizeof(float), cudaMemcpyHostToDevice));

  // start timer (only time the calculation of the max value,
  // including the list will make the time increase.
  maxTimer.start();
  
  // preform CUDA parallelism
  maxKernel<<<block_size,tpb>>>(d_list);
  
  // stop time
  maxTimer.stop();
  
  // return maximum value to user
  printf("\nThe maximum value found is: %lf\n",list[0]);
  
  // show time needed for the calculation
  cout << maxTimer;
    
  // copy memory back from device
  checkCudaCall(cudaMemcpy(list, d_list, sizeof(float)*length, cudaMemcpyDeviceToHost));

  // Free device mem.
  checkCudaCall(cudaFree(d_list));
  checkCudaCall(cudaFree(d_max)); 
}

