/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#include "simulate.h"

typedef struct{
	int i_min;
	int i_max;
} i_range_t;


/* Add any global variables you may need. */
double *gl_old_array;
double *gl_current_array;
double *gl_next_array;

int nThreadsUnfinished;
int numThreads;
pthread_cond_t threadsDone = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

/* Add any functions you may need (like a worker) here. */
void *compute(void *p){
	// bereken A voor gegeven i's -> wachten tot alle threads klaar zijn
	// --> berekenen A voor gegeven i's voor t+1 --> etc
	i_range_t *range = (i_range_t *) p;
	
	int t;
	// TODO: t_max ipv 10
	for(t = 0; t < 10; t++){
		printf("Hello world! (%d) Range: %d-%d\n", t, range->i_min, range->i_max);
		
		// Calculate Ai_min, t t/m Ai_max, t here
		
		pthread_mutex_lock(&lock);
		nThreadsUnfinished--;
		if(nThreadsUnfinished == 0){
			// Signal waiting threads, this t is complete.
			pthread_cond_broadcast(&threadsDone);
			// Reset num unfinished threads
			nThreadsUnfinished = numThreads;
			
			//TODO: swap buffers
		}else{
			// Wait till all threads have completed this t
			pthread_cond_wait(&threadsDone, &lock);
		}
		pthread_mutex_unlock(&lock);
	}
	return NULL;
}

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
		double *old_array, double *current_array, double *next_array){
	gl_old_array = old_array;
	gl_current_array = current_array;
	gl_next_array = next_array;
	numThreads = num_threads;
	
	int nThread;
	int iPerThread = i_max / num_threads;
	
	// TODO: vervangen door malloc
	pthread_t threadIds[num_threads];
	i_range_t iRanges[num_threads];
	
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
	
	nThreadsUnfinished = num_threads;		
	
	for(nThread = 0; nThread < num_threads; nThread++){
		// Set the i range the thread should calculate
		iRanges[nThread].i_min = iPerThread * nThread;
		
		if(nThread + 1 == num_threads){
			iRanges[nThread].i_max = i_max - 1;
		}else{
			iRanges[nThread].i_max = iPerThread * (nThread + 1) - 1;
		}
		
		// TODO: check return waarde
		pthread_create(&threadIds[nThread], &attr, &compute, &iRanges[nThread]);
	}
	
	void *result;
	for(nThread = 0; nThread < num_threads; nThread++){
		pthread_join(threadIds[nThread], &result);
	}
	
	pthread_attr_destroy(&attr);

	/*
	* After each timestep, you should swap the buffers around. Watch out none
	* of the threads actually use the buffers at that time.
	*/


	/* You should return a pointer to the array with the final results. */
	return next_array;
}
