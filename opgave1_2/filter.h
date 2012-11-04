#ifndef FILTER_H
#define FILTER_H

#include "queue.h"

pthread_t *startFilter(queue_t *queue);

void *filter(void *p);

#endif
