#ifndef FILTER_H
#define FILTER_H

typedef struct{
	queue_t old_queue;
	queue_t next_queue;
	unsigned long long filter_value;
	pthread_mutex_t *lock;
} filter_t;

filter_t *newFilter(unsigned long long value, queue_t *old);

void *filter(void *p);

#endif
