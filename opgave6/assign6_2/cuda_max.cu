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


// TODO create some function that adds values.
__global__ void maxKernel() {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	
}

// TODO create working compute function that returns the max value of an array
float *computeMaxCuda(int length){
    
  float* d_list = NULL;
  float* d_max = NULL;
  float* maxVal = NULL;
  int tpb = 128;
  float list[length];
  
  srand(time(NULL));
  //TODO make this run in parallel
  for(int i = 0; i< length); i++)
    list[i] = (float)rand()/((float)RAND_MAX/FLT_MAX);
  
	// Alloc space on the device.
	// Is this the right amount?
	checkCudaCall(cudaMalloc((void **) &d_list, length * sizeof(float)));
	checkCudaCall(cudaMalloc((void **) &d_max, sizeof(float)));
	
	// TODO make the right call
  maxKernel<<<(int) ceil((double) length / (double) tpb), tpb>>>();

  // copy resulting max back to main memory
  checkCudaCall(cudaMemcpy(maxVal, d_max, sizeof(float), cudaMemcpyDeviceToHost));

	// Free device mem.
	checkCudaCall(cudaFree(d_list));
	checkCudaCall(cudaFree(d_max));
	
	return maxVal;
}
