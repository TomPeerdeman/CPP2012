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

// TODO create compute function that returns the max value of an array
float *computeMaxCuda(int length){
  srand(time(NULL));
  float list[length];
  //TODO make this run in parallel
  for(int i = 0; i< length); i++)
    list[i] = (float)rand()/((float)RAND_MAX/FLT_MAX);
    
  float *dOld
  int tpb = 128;
    
	// Alloc space on the device.
	// Is this the right amount?
	checkCudaCall(cudaMalloc((void **) &dOld, length * sizeof(float)));
	
	// TODO make the right call
  maxKernel<<<(int) ceil((double) length / (double) tpb), tpb>>>();
	
	// Free device mem.
	checkCudaCall(cudaFree(dOld));
	
	return maxVal;
}
