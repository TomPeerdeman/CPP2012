#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <pthread.h>

#include "queue.h"
#include "filter.h"

pthread_t *startFilter(queue_t *queue){
	pthread_t *threadId = malloc(sizeof(pthread_t));
	
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	// Make system threads, so we can utilize multiple cores.
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	// Set the stack size to minimum so we can create more threads.
	pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN);
	
	int err = pthread_create(threadId, &attr, &filter, queue);
	if(err){
		printf("Thread create error %d\n", err);
		exit(1);
	}
	
	pthread_attr_destroy(&attr);
	
	return threadId;
}

void *filter(void *p){
	queue_t *inQueue = (queue_t *) p;
	queue_t *outQueue = NULL;
	unsigned long long val = 0;
	unsigned long long divider = 2;
	pthread_t *nextThread = NULL;

	while(1){
		val = dequeue(inQueue);
		
		if(outQueue != NULL){
			// If the value 1 is given, start with terminating this filter.
			if(val == 1){
				enqueue(outQueue, val);
				void *result;
				pthread_join(*nextThread, &result);
				free(nextThread);
				freeQueue(outQueue);
				break;
			}
			
			/*
			 * If the value is not a multiple of this filter's divider
			 * send it to the next filter.
			 */
			if(val % divider != 0){
				enqueue(outQueue, val);
			}

		}else{
			/*
			 * No next filter yet, the first value in the queue is this
			 * filter's divider.
			 */
			divider = val;
			if(val == 1){
				break;
			}
			printf("%llu\n", divider);
			
			/*
			 * Search for the next non multiple of the divider, the
			 * value found will be the divider of the next filter.
			 * If a 1 (terminate) is found, we don't have to build the
			 * next filter.
			 */
			do{
				val = dequeue(inQueue);
			}while(val != 1 && val % divider == 0);

			if(val != 1){
				outQueue = newQueue();
				if(outQueue == NULL){
					break;
				}
				enqueue(outQueue, val);
				nextThread = startFilter(outQueue);
			}else{
				break;
			}
		}
	}
	
	return NULL;
}
