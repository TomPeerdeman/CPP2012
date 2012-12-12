#include <math.h>
#include <cstdio>
#include <float.h>
#include <time.h>
#include <cstdlib>
#include <string.h>

#include "cuda_max.h"
#include "seq_max.h"

int main(int argc, char **argv){
srand(time(NULL));

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
  
  float list[length];
  
  int block_size = (int) ceil((double) length / (double) tpb);
  // make a list of floats
  for(int i = 0; i< length; i++){
    list[i] = (float)rand()/((float)RAND_MAX/FLT_MAX);
	printf("List[%d]: %f\n", i, list[i]);
  }
  
  float maxSeq = computeMaxSeq(length, list);
  printf("Seq max:   %f\n", maxSeq);
  float maxCUDA = computeMaxCuda(length, block_size, tpb, list);  
  printf("CUDA max: %f\n", maxCUDA);

  
  printf("Difference seq/CUDA: %f\n", (maxSeq - maxCUDA));
  return 0;
}
