/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>
#include <errno.h>

#include "simulate.h"

#define SPATIAL_IMPACT 0.2

typedef struct{
	int i_min;
	int i_max;
} i_range_t;


/* Add any global variables you may need. */
double *gl_old_array;
double *gl_current_array;
double *gl_next_array;
int gl_t_max;
int gl_i_max;

int nThreadsUnfinished;
int numThreads;
pthread_cond_t threadsDone = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

/* Add any functions you may need (like a worker) here. */
void *compute(void *p){
	i_range_t *range = (i_range_t *) p;
	
	int t;
	int i;
	double *temp;
	
	for(t = 0; t < gl_t_max; t++){		
		// Calculate Ai_min, t up to and including Ai_max, t here
		for(i = range->i_min; i <= range->i_max; i++){
			gl_next_array[i] = 2 * gl_current_array[i] - gl_old_array[i]
				+ SPATIAL_IMPACT * (
					((i > 1) ? gl_current_array[i - 1] : 0) - (
						2 * gl_current_array[i] - 
						((i < gl_i_max - 1) ? gl_current_array[i + 1]: 0)
					)
				);
		}
		
		pthread_mutex_lock(&lock);
		nThreadsUnfinished--;
		if(nThreadsUnfinished == 0){
			/* Signal waiting threads, this t is complete.
			 * We can signal before the buffers are rotated because
			 * the waiting threads need to aquire the lock first
			 * that this thread holds to continue.
			 */
			pthread_cond_broadcast(&threadsDone);
			// Reset num unfinished threads
			nThreadsUnfinished = numThreads;
			
			// Rotate buffers
			temp = gl_old_array;
			gl_old_array = gl_current_array;
			gl_current_array = gl_next_array;
			gl_next_array = temp;
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
double *simulate(const int i_max, const int t_max, int num_threads,
		double *old_array, double *current_array, double *next_array){
	gl_old_array = old_array;
	gl_current_array = current_array;
	gl_next_array = next_array;
	gl_t_max = t_max;
	gl_i_max = i_max;
	numThreads = num_threads;
	
	int nThread;
	int iPerThread = i_max / num_threads;

	pthread_t *threadIds;
	// locate memory for the threadIds if possible, else break
	threadIds = malloc(num_threads*sizeof(pthread_t));
	if(threadIds == NULL){
		perror("Error locating memory! \n");
		exit(1);
	}
	
	i_range_t *iRanges;
	// locate memory for the iRanges if possible, else break
	iRanges = malloc(num_threads*sizeof(i_range_t));
	if(iRanges == NULL){
		free(threadIds);
		perror("Error locating memory! \n");
		exit(1);
	}
	
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
		int err = pthread_create(&threadIds[nThread], &attr, &compute, 
			&iRanges[nThread]);
		if(err){
			switch(err){
				case EAGAIN:
					printf("Error creating thread %d/%d: Insufficient" 
						"resources (%d)\n", nThread, num_threads,
						err);
					break;
				case EPERM:
					printf("Error creating thread %d/%d:"
						"No permission to set the scheduling "
						"policy and parameters specified (%d)\n",
						nThread, num_threads, err);
					break;
				default:
					printf("Error creating thread %d/%d: error %d\n",
						nThread, num_threads, err);
			}
			num_threads = nThread;
			break;
		}			
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
	
	// free memory when complete
	free(threadIds);
	free(iRanges);

	/* You should return a pointer to the array with the final results. */
	return gl_current_array;
}
