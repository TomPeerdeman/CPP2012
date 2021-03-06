#ifndef QUEUE_H
#define QUEUE_H

#define BUFFER_SIZE 100

typedef struct{
	unsigned long long *queue;
	int readPtr;
	int writePtr;
	int fill;
	pthread_mutex_t lock;
	pthread_cond_t full;
	pthread_cond_t empty;
} queue_t;

queue_t *newQueue(void);

void freeQueue(queue_t *queue);

void enqueue(queue_t *queue, unsigned long long v);

unsigned long long dequeue(queue_t *queue);

#endif
