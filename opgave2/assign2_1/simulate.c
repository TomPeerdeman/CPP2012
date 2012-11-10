/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"
#include "mpi.h"

#define SPATIAL_IMPACT 0.2


/* Add any global variables you may need. */


/* Add any functions you may need (like a worker) here. */


/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int iPerTask, const int t_max, double *old_array,
        double *current_array, double *next_array, int my_rank, int num_tasks)
{
    int min_i, max_i;
    int i, t;
    
    //initialize i domains, considering halo cells if there are any.
    if(my_rank == 0 && num_tasks == 1){
      min_i = 0;
      max_i = iPerTask;    
    }
    else if(my_rank == 0){
      min_i = 0;
      max_i = iPerTask + 1;
    }
    else if(my_rank == num_tasks-1){
      min_i = (num_tasks - 1) * iPerTask - 1;
      max_i = num_tasks * iPerTask;
    }
    else{
      min_i = my_rank * iPerTask - 1;
      max_i = (my_rank + 1) * iPerTask + 1;
    }
    
    printf("This is my rank: %d out of %d tasks", my_rank, num_tasks);

    /*
     * Your implementation should go here.
     */
    for(t = 0; t < t_max; t++){		
		// Calculate Ai_min, t up to and including Ai_max, t here
		  for(i = min_i; i <= i_max; i++){
			  /*gl_next_array[i] = 2 * gl_current_array[i] - gl_old_array[i]
				  + SPATIAL_IMPACT * (
					  ((i > 1) ? gl_current_array[i - 1] : 0) - (
						  2 * gl_current_array[i] - 
						  ((i < gl_i_max - 1) ? gl_current_array[i + 1]: 0)
					  )
				  );*/
		  }
		}
    

    /* You should return a pointer to the array with the final results. */
    return current_array;
}
