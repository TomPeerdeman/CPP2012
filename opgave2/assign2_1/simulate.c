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
void calculate(double *old, double *cur, double *next, int i){
  next[i] = 2.0 * cur[i] - old[i] + SPATIAL_IMPACT * (
      (cur[i - 1] - (2.0 * cur[i] - cur[i + 1]))
    );
}

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
    double discard;
    MPI_Status status;
    
    //initialize i domains, considering halo cells
    min_i = 1;
    max_i = iPerTask;
  
    /*
     * ALL nodes are assigned arrays of length iPerTask + 2 elements.
     * For all nodes the data starts at 1, the last element of data is at index
     * iPerTask. The return array looks the same as the input array.
     */
        
    //Send only if left exists.
    if(my_rank != 0){
      MPI_Send(&current_array[min_i], 1, MPI_DOUBLE, my_rank-1, 1, MPI_COMM_WORLD);    
    }
    
    //Send only if right exists.
    if(my_rank != num_tasks-1){
      MPI_Send(&current_array[max_i], 1, MPI_DOUBLE, my_rank+1, 1, MPI_COMM_WORLD);
    }
  
    for(t = 0; t < t_max; t++){    
    // Calculate Ai_min, t up to and including Ai_max, t here
    for(i = min_i + 1; i < max_i; i++){
      calculate(old_array, current_array, next_array, i);
    }
    
    if(my_rank != num_tasks-1){
      MPI_Recv(&current_array[max_i+1], 1, MPI_DOUBLE, my_rank+1, 1, MPI_COMM_WORLD, &status);

      calculate(old_array, current_array, next_array, max_i);

      MPI_Send(&next_array[max_i], 1, MPI_DOUBLE, my_rank+1, 1, MPI_COMM_WORLD);
    }else{
      next_array[max_i] = 0.0;
    }
      
    //Send only if left exists.
    if(my_rank != 0){
      MPI_Recv(&current_array[min_i-1], 1, MPI_DOUBLE, my_rank-1, 1, MPI_COMM_WORLD, &status);

      calculate(old_array, current_array, next_array, min_i);  
      
      MPI_Send(&next_array[min_i], 1, MPI_DOUBLE, my_rank-1, 1, MPI_COMM_WORLD);
    }else{
      next_array[min_i] = 0.0;  
    }      
    
    // Rotate buffers
    temp = old_array;
    old_array = current_array;
    current_array = next_array;
    next_array = temp;
  }
    
    if(my_rank != num_tasks-1){
      MPI_Recv(&discard, 1, MPI_DOUBLE, my_rank+1, 1, MPI_COMM_WORLD, &status);
    }
    if(my_rank != 0){
      MPI_Recv(&discard, 1, MPI_DOUBLE, my_rank-1, 1, MPI_COMM_WORLD, &status);
    }
  
    /* You should return a pointer to the array with the final results. */
    return current_array;
}
