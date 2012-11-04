#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "filter.h"

pthread_t *startFilter(queue_t *queue){
	pthread_t *threadId = malloc(sizeof(pthread_t));
	
	// TODO: Set attributes
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	//pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN);
	
	pthread_create(threadId, &attr, &filter, queue);
	
	pthread_attr_destroy(&attr);
	
	return threadId;
}

void *filter(void *p){
	queue_t *inQueue = (queue_t *) p;
	queue_t *outQueue = NULL;
	unsigned long long val;
	unsigned long long divider;
	pthread_t *nextThread;

	while(1){
		val = dequeue(inQueue);
		
		if(outQueue != NULL){
			// if the value 1 is given, start with terminating this filter.
			if(val == 1){
				enqueue(outQueue, val);
				printf("i have to die (%llu):(\n", divider);
				void *result;
				pthread_join(*nextThread, &result);
				free(nextThread);
				freeQueue(outQueue);
				break;
			}
			
			if(val % divider != 0){
				enqueue(outQueue, val);
			}

		}else{
			divider = val;
			if(val == 1){
				break;
			}
			
			printf("%llu\n", divider);
			do{
				val = dequeue(inQueue);
			}while(val != 1 && val % divider == 0);

			if(val != 1){
				// TODO: check null pointer if failed
				outQueue = newQueue();
				enqueue(outQueue, val);
				nextThread = startFilter(outQueue);
			}else{
				break;
			}
		}
	}
	
	
	printf("%llu done.\n", divider);
	return NULL;
}
