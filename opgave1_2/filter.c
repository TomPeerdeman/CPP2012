#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "filter.h"

void startFilter(queue_t *queue){
	pthread_t threadId;
	
	// TODO: Set attributes
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN);
	
	pthread_create(&threadId, &attr, &filter, queue);
	
	pthread_attr_destroy(&attr);
}

void *filter(void *p){
	queue_t *inQueue = (queue_t *) p;
	queue_t *outQueue = NULL;
	unsigned long long val;
	unsigned long long divider;

	while(1){
		val = dequeue(inQueue);
		
		if(outQueue != NULL){
			if(val % divider != 0){
				enqueue(outQueue, val);
			}
		}else{
			divider = val;
			printf("%llu\n", divider);
			do{
				val = dequeue(inQueue);
			}while(val % divider == 0);
			// TODO: check null pointer if failed
			outQueue = newQueue();
			enqueue(outQueue, val);
			startFilter(outQueue);
		}
	}
}
