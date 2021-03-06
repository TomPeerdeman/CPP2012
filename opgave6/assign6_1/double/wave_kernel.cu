#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "timer.h"

#define SPATIAL_IMPACT 0.2 

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

__global__ void waveKernel(unsigned int n, double* old, double* cur, double* next) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	
	// Don't calculate the borders or out of the borders.
	if(i > 0 && i < n){
		next[i] = 2.0 * cur[i] - old[i] +
            SPATIAL_IMPACT * ((cur[i - 1] - (2.0 * cur[i] - cur[i + 1])));
	}
}

double *computeWaveCuda(int i_max, int t_max, int tpb, double *hOld, double *hCur, double *hNext){
	double *dOld, *dCur, *dNext, *tmp;
	
	// Alloc space on the device.
	checkCudaCall(cudaMalloc((void **) &dOld, i_max * sizeof(double)));
	checkCudaCall(cudaMalloc((void **) &dCur, i_max * sizeof(double)));
	checkCudaCall(cudaMalloc((void **) &dNext, i_max * sizeof(double)));
	
	// Copy from main mem to device mem.
	checkCudaCall(cudaMemcpy(dOld, hOld, i_max*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaCall(cudaMemcpy(dCur, hCur, i_max*sizeof(double), cudaMemcpyHostToDevice));
	
	timer waveTimer("Wave timer");
	waveTimer.start();
	int t;
	for(t = 0; t < t_max; t++){
		// Start the computation for time = t.
		waveKernel<<<(int) ceil((double) i_max / (double) tpb), tpb>>>(i_max, dOld, dCur, dNext);
		
		checkCudaCall(cudaGetLastError());
		
		// Rotate buffers.
		tmp = dOld;
		dOld = dCur;
		dCur = dNext;
		dNext = tmp;
		
		tmp = hOld;
		hOld = hCur;
		hCur = hNext;
		hNext = tmp;
	}
	waveTimer.stop();
	
	cout << waveTimer;
	
	// Copy back the result from device mem to main mem.
	checkCudaCall(cudaMemcpy(hCur, dCur, i_max * sizeof(double), cudaMemcpyDeviceToHost));
	
	// Free device mem.
	checkCudaCall(cudaFree(dOld));
	checkCudaCall(cudaFree(dCur));
	checkCudaCall(cudaFree(dNext));
	
	return hCur;
}
