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
    double *temp;
    
    //initialize i domains, considering halo cells if there are any.
    if(my_rank == num_tasks-1){
      min_i = 0;
      max_i = sizeof(current_array)/sizeof(double);
    }
    else{
      min_i = 1;
      max_i = sizeof(current_array)/sizeof(double) - 1;
    }
    
    printf("This is my rank: %d out of %d tasks", my_rank, num_tasks);

    /*
     * Your implementation should go here.
     */
     
    // send(left_neighbour, cur[1])
    // send(right_neighbour, cur[size])
    for(t = 0; t < t_max; t++){		
		// Calculate Ai_min, t up to and including Ai_max, t here
		  for(i = min_i; i < max_i; i++){
			  next_array[i] = 2 * current_array[i] - old_array[i]
				  + SPATIAL_IMPACT * (
					  ((i > 1) ? current_array[i - 1] : 0) - (
						  2 * current_array[i] - 
						  ((i < iPerTask - 1) ? current_array[i + 1]: 0)
					  )
				  );
		  }
		  /*current_array[max_i+1] = receive(right_neighbour);
		  next_array[max_i] = 2 * current_array[max_i] - old_array[max_i]
				  + SPATIAL_IMPACT * (
					  ((max_i > 1) ? current_array[max_i - 1] : 0) - (
						  2 * current_array[max_i] - 
						  ((max_i < iPerTask - 1) ? current_array[max_i + 1]: 0)
					  )
				  ); 
      send(right_neighbour, next_array[size])
		  
		  current_array[0] = receive(left_neighbour);
		  next_array[1] = 2 * current_array[1] - old_array[1]
				  - (2 * current_array[1] - current_array[2])
				  ); 
      send(left_neighbour, next_array[1])
		  */
		  
		  
		  // Rotate buffers
			temp = old_array;
			old_array = current_array;
			current_array = next_array;
			next_array = temp;
		}
    //discard = receive(right_neighbour);
    //discard = receive(left_neighbour);

    /* You should return a pointer to the array with the final results. */
    return current_array;
}