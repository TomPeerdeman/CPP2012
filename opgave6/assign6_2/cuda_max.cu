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
    
    
	// Alloc space on the device. TODO alloc right amount
	checkCudaCall(cudaMalloc((void **) &dOld, i_max * sizeof(float)));
	
	// Copy from main mem to device mem. TODO alloc right amount
	checkCudaCall(cudaMemcpy(dOld, hOld, i_max*sizeof(float), cudaMemcpyHostToDevice));
	checkCudaCall(cudaMemcpy(dCur, hCur, i_max*sizeof(float), cudaMemcpyHostToDevice));
	
	// TODO make the right call
  maxKernel<<<(int) ceil((double) i_max / (double) tpb), tpb>>>(i_max, dOld, dCur, dNext);
	
	// Copy back the result from device mem to main mem. TODO copy right amount back
	checkCudaCall(cudaMemcpy(hCur, dCur, i_max * sizeof(float), cudaMemcpyDeviceToHost));
	
	// Free device mem.
	checkCudaCall(cudaFree(dOld));
	
	return maxVal;
}
