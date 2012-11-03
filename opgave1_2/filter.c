#include <stdlib.h>
#include <stdio.h>

#include "queue.h"
#include "filter.h"

filter_t *newFilter(unsigned long long value, queue_t *old){
	filter_t *filter = malloc(sizeof(filter_t));
	if(filter == NULL){
		perror("Error: memory allocation");
		return NULL;
	}

  filter->filter_value = value;
  filter->old_queue = old;
  //no new queue yet, only after the filter finds a prime.
  filter->next_queue = NULL;
  
	pthread_mutex_init(filter->lock, NULL);
  
  return filter;
}

void *filter(void *p){

  /*
   *if(filter->next_queue == NULL)
   *  next_queue = newqueue();
   */
}
