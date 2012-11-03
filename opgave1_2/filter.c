#include <stdlib.h>
#include <stdio.h>

#include "filter.h"

void startFilter(queue_t *queue){
	pthread_t threadId;
	
	// TODO: Set attributes
	
	pthread_create(&threadId, NULL, &filter, queue);
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
			printf("%llu", divider);
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
