#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>

#define BUFFER_SIZE 100

typedef struct{
	unsigned long long *queue;
	int readPtr;
	int writePtr;
	int fill;
	filter_t next_filter;
	pthread_mutex_t *lock;
	pthread_cond_t *full;
	pthread_cond_t *empty;
} queue_t;

queue_t *newQueue(unsigned long long filter_value);

void freeQueue(queue_t *queue);

void enqueue(queue_t *queue, unsigned long long v);

unsigned long long dequeue(queue_t *queue);

#endif
