#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>

typedef struct{
	int known_nr;
} filter;

int main(int argc, char *argv[]){
  //pthread_t *threads = NULL;
  int upperbound;
  int counter = 2;
  
  /* 
   * You can give an argument telling the program you only want prime numbers
   * below a given number. If you dont give this argument, prime numbers will
   * be calculated until the maximum value for an integer is achieved.
   */
  if(argc == 2 && atoi(argv[1])>2){
    upperbound = atoi(argv[1]);
  }
  else{
    upperbound = INT_MAX;
  }
  printf("Upperbound for primes: %d\n",upperbound);
  

  
  return 0;
}

void *compute(void *p){

}
