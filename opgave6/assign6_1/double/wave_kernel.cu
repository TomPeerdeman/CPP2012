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
	
	
	printf("dOld: %p\n", dOld);
	printf("dCur: %p\n", dCur);
	printf("dNext: %p\n", dNext);

	printf("hOld: %p\n", hOld);
	printf("hCur: %p\n", hCur);
	printf("hNext: %p\n", hNext);
	
	
	// Copy from main mem to device mem.
	checkCudaCall(cudaMemcpy(dOld, hOld, i_max*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaCall(cudaMemcpy(dCur, hCur, i_max*sizeof(double), cudaMemcpyHostToDevice));
	
	cout << "No segfault in copy!" << endl;
	cout << "float: " << sizeof(float) << " double: " << sizeof(double)  << endl;
	
	cout << "HOld[0]: " << hOld[0] << endl;	
	cout << "DOld[0]: " << dOld[0] << endl;
	cout << "HOld[10]: " << hOld[10] << endl;
	cout << "DOld[10]: " << dOld[10] << endl;
	
	
	
	timer waveTimer("Wave timer");
	waveTimer.start();
	int t;
	for(t = 0; t < t_max; t++){
		// Start the computation for time = t.
		waveKernel<<<i_max / tpb, tpb>>>(i_max, dOld, dCur, dNext);
		
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

//~ void vectorAddCuda(int n, float* a, float* b, float* result) {
    //~ int threadBlockSize = 512;

    //~ // allocate the vectors on the GPU
    //~ float* deviceA = NULL;
    //~ checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(float)));
    //~ if (deviceA == NULL) {
        //~ cout << "could not allocate memory!" << endl;
        //~ return;
    //~ }
    //~ float* deviceB = NULL;
    //~ checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(float)));
    //~ if (deviceB == NULL) {
        //~ checkCudaCall(cudaFree(deviceA));
        //~ cout << "could not allocate memory!" << endl;
        //~ return;
    //~ }
    //~ float* deviceResult = NULL;
    //~ checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(float)));
    //~ if (deviceResult == NULL) {
        //~ checkCudaCall(cudaFree(deviceA));
        //~ checkCudaCall(cudaFree(deviceB));
        //~ cout << "could not allocate memory!" << endl;
        //~ return;
    //~ }

    //~ cudaEvent_t start, stop;
    //~ cudaEventCreate(&start);
    //~ cudaEventCreate(&stop);

    //~ // copy the original vectors to the GPU
    //~ checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(float), cudaMemcpyHostToDevice));
    //~ checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(float), cudaMemcpyHostToDevice));

    //~ // execute kernel
    //~ cudaEventRecord(start, 0);
    //~ vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceA, deviceB, deviceResult);
    //~ cudaEventRecord(stop, 0);

    //~ // check whether the kernel invocation was successful
    //~ checkCudaCall(cudaGetLastError());

    //~ // copy result back
    //~ checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(float), cudaMemcpyDeviceToHost));

    //~ checkCudaCall(cudaFree(deviceA));
    //~ checkCudaCall(cudaFree(deviceB));
    //~ checkCudaCall(cudaFree(deviceResult));

    //~ // print the time the kernel invocation took, without the copies!
    //~ float elapsedTime;
    //~ cudaEventElapsedTime(&elapsedTime, start, stop);
    
    //~ cout << "kernel invocation took " << elapsedTime << " milliseconds" << endl;
//~ }


//~ int main(int argc, char* argv[]) {
    //~ int n = 65536;
    //~ timer vectorAddTimer("vector add timer");
    //~ float* a = new float[n];
    //~ float* b = new float[n];
    //~ float* result = new float[n];

    //~ // initialize the vectors.
    //~ for(int i=0; i<n; i++) {
        //~ a[i] = i;
        //~ b[i] = i;
    //~ }

    //~ vectorAddTimer.start();
    //~ vectorAddCuda(n, a, b, result);
    //~ vectorAddTimer.stop();

    //~ cout << vectorAddTimer;

    //~ // verify the resuls
    //~ for(int i=0; i<n; i++) {
        //~ if(result[i] != 2*i) {
            //~ cout << "error in results! Element " << i << " is " << result[i] << ", but should be " << (2*i) << endl;
            //~ exit(1);
        //~ }
    //~ }
    //~ cout << "results OK!" << endl;
            
    //~ delete[] a;
    //~ delete[] b;
    //~ delete[] result;
    
    //~ return 0;
//~ }
