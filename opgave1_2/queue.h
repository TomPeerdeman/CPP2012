#ifndef QUEUE_H
#define QUEUE_H

#include <pthread.h>

typedef struct{
	unsigned long long *queue;
	int readPtr;
	int writePtr;
	int fill;
	pthread_mutex_t lock;
	pthread_cond_t full;
	pthread_cond_t empty;
} queue_t;

void enqueue(queue_t *queue, unsigned long long v);

unsigned long long dequeue(queue_t *queue);

#endif
