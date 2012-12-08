#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>

#include "cuda_max.h"

int main(int argc, char **argv){
	if(argc < 2){
	  printf("Usage: %s <number of floats in the list>\n", argv[0]);
	  return 1;
	}
	int input = atoi(argv[1]);
	if(input == 0){
	  printf("The amount of floats in the list must be a number!\n");
	  return 1;
	}
	
	
  ret = computeWaveCuda(input);
  printf("The maximum value found is: %lf",ret);
  
  return 0;
}
