#include <stdlib.h>
#include <stdio.h>

#include "queue.h"

//initialize a queue
queue_t *newQueue(){
	queue_t *queue = malloc(sizeof(queue_t));
	if(queue == NULL){
		perror("Error: memory allocation");
		return NULL;
	}
	
	queue->queue = malloc(BUFFER_SIZE * sizeof(unsigned long long));
	if(queue->queue == NULL){
		freeQueue(queue);
		perror("Error: memory allocation");
		return NULL;
	}
	
	queue->readPtr = 0;
	queue->writePtr = 0;
	queue->fill = 0;
	
	// TODO: check return waarde?
	pthread_mutex_init(queue->lock, NULL);
	
	pthread_cond_init(queue->full, NULL);
	pthread_cond_init(queue->empty, NULL);
	
	//TODO: return value
}

//clean up the queue
void freeQueue(queue_t *queue){
	if(queue != NULL){
		if(queue->queue != NULL){
			free(queue->queue);
		}
		
		if(queue->lock != NULL){
			pthread_mutex_destroy(queue->lock);
		}
		
		if(queue->full != NULL){
			pthread_cond_destroy(queue->full);
		}
		
		if(queue->empty != NULL){
			pthread_cond_destroy(queue->empty);
		}
		
		free(queue);
	}
}

// add a value to the queue
void enqueue(queue_t *queue, unsigned long long val){
	pthread_mutex_lock(queue->lock);
	// Queue full, sleep till a space is available
	while(queue->fill == BUFFER_SIZE){
		pthread_cond_wait(queue->full, queue->lock);
	}
	
	queue->queue[queue->writePtr++] = val;
	queue->writePtr %= BUFFER_SIZE;
	
	queue->fill++;
	
	// Signal sleeping consumer
	pthread_cond_signal(queue->empty);
	
	pthread_mutex_unlock(queue->lock);
}

// remove the first number from the queue
unsigned long long dequeue(queue_t *queue){
	unsigned long long val;
	
	pthread_mutex_lock(queue->lock);
	// Queue empty, sleep for item to arrive
	while(queue->fill == 0){
		pthread_cond_wait(queue->empty, queue->lock);
	}
	
	val = queue->queue[queue->readPtr++];
	queue->readPtr %= BUFFER_SIZE;
	
	queue->fill--;
	
	// Signal sleeping producer
	pthread_cond_signal(queue->full);
	
	pthread_mutex_unlock(queue->lock);
	
	//TODO: return value
}
