#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <pthread.h>

#include "queue.h"
#include "filter.h"

int main(int argc, char *argv[]){
	unsigned long long upperbound;
	unsigned long long counter;

	/* 
	 * You can give an argument telling the program you only want prime numbers
	 * below a given number. If you dont give this argument, prime numbers will
	 * be calculated until the maximum value for an integer is achieved.
	 */
	if(argc == 2 && atoi(argv[1])>2){
		upperbound = atoi(argv[1]);
	}
	else{
		upperbound = LLONG_MAX;
	}
	printf("Upperbound for primes: %lld\n",upperbound);
	// Create the first queue.
	queue_t *first_queue = newQueue();

	pthread_t *first_thread = startFilter(first_queue);
	printf("Started first filter\n");

	/*
	 * Add all the natural numbers starting with two to the first queue,
	 * if the queue is full the number waits to be added.
	 */
	for(counter = 2; counter < upperbound; counter++){
		enqueue(first_queue, counter);
	}
	printf("Done filling \n");
	
	/*
	 * Send the number one, this way the filters will know they have 
	 * to terminate.
	 */
	enqueue(first_queue, 1LL);
	
	// Join first thread, which joins the second thread etc.
	void *result;
	pthread_join(*first_thread, &result);
	
	freeQueue(first_queue);

	return 0;
}
