#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>

#include "cuda_max.h"

int main(int argc, char **argv){
  float* ret;
	if(argc < 3){
	  printf("Usage: %s <number of floats in the list> <Threads per block>\n", argv[0]);
	  return 1;
	}
	int length = atoi(argv[1]);
	if(input == 0){
	  printf("The amount of floats in the list must be a number!\n");
	  return 1;
	}
	
	int tpb = atoi(argv[2]);
	if(tpb == 0){
	  printf("The amount of threads per block must be a number!\n");
	  return 1;
	}
	int block_size = (int) ceil((double) length / (double) tpb);
	
  ret = computeMaxCuda(length, block_size, tpb);
  printf("The maximum value found is: %lf",ret);
  
  return 0;
}
