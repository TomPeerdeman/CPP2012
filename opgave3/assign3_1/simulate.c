/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#include "simulate.h"

#define SPATIAL_IMPACT 0.2 


/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    /*
     * Your implementation should go here.
     */
    omp_set_num_threads(num_threads);
    
    double *temp;
    int t, i;
    int x = 0;
    for(t = 0;t < t_max; t++)
    {
        # pragma omp parallel for private(i) firstprivate(x)
        for(i = 0; i < i_max; i++)
        {
          if(x==0)
            //printf("%d,%d\n",i,omp_get_thread_num());
          x = 1;
          
            next_array[i] = 2.0 * current_array[i] - old_array[i] +
            SPATIAL_IMPACT * ((current_array[i - 1] -
             (2.0 * current_array[i] - current_array[i + 1])));
        }
        temp = old_array;
        old_array = current_array;
        current_array = next_array;
        next_array = temp;
            
    }
    return current_array;
}
