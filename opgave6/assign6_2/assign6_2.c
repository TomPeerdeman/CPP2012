#include <math.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>

#include "cuda_max.h"

int main(int argc, char **argv){
  if(argc < 3){
    printf("Usage: %s <number of floats in the list> <Threads per block>\n", argv[0]);
    return 1;
  }
  int length = atoi(argv[1]);
  if(length == 0){
    printf("The amount of floats in the list must be a number!\n");
    return 1;
  }
  
  int tpb = atoi(argv[2]);
  if(tpb == 0){
    printf("The amount of threads per block must be a number!\n");
    return 1;
  }
  int block_size = (int) ceil((double) length / (double) tpb);
  
  computeMaxCuda(length, block_size, tpb);  
  return 0;
}
