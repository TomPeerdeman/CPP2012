#ifndef FILTER_H
#define FILTER_H

pthread_t *startFilter(queue_t *queue);

void *filter(void *p);

#endif
