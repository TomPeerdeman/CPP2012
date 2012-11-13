/*
 * assign1_1.c
 *
 * Contains code for setting up and finishing the simulation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "file.h"
#include "timer.h"
#include "simulate.h"
#include "mpi.h"

typedef double (*func_t)(double x);

/*
 * Simple gauss with mu=0, sigma^1=1
 */
double gauss(double x){
    return exp((-1 * x * x) / 2);
}


/*
 * Fills a given array with samples of a given function. This is used to fill
 * the initial arrays with some starting data, to run the simulation on.
 *
 * The first sample is placed at array index `offset'. `range' samples are
 * taken, so your array should be able to store at least offset+range doubles.
 * The function `f' is sampled `range' times between `sample_start' and
 * `sample_end'.
 */
void fill(double *array, int offset, int range, double sample_start,
        double sample_end, func_t f){
    int i;
    float dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++) {
        array[i + offset] = f(sample_start + i * dx);
    }
}


int main(int argc, char *argv[]){
  double *old, *current, *next, *ret;
  int rc , num_tasks , my_rank, nitems, t_max;
  
  // Setting up MPI.
  rc = MPI_Init(&argc, &argv);
  if(rc != MPI_SUCCESS){
    fprintf(stderr, "Unable to set up MPI");
    // Abort MPI runtime.
    MPI_Abort(MPI_COMM_WORLD, rc);
  }
  MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if(my_rank == 0){
    int i_max, iPerTask;
    int t, low, high;
    double time;
    int *startPtrs;
    
    /* Parse commandline args */
    if (argc < 3) {
      printf("Usage: %s i_max t_max num_threads [initial_data]\n", argv[0]);
      printf(" - i_max: number of discrete amplitude points, should be >2\n");
      printf(" - t_max: number of discrete timesteps, should be >=1\n");
      printf(" - num_threads: number of threads to use for simulation, "
          "should be >=1\n");
      printf(" - initial_data: select what data should be used for the first "
          "two generation.\n");
      printf("   Available options are:\n");
      printf("    * sin: one period of the sinus function at the start.\n");
      printf("    * sinfull: entire data is filled with the sinus.\n");
      printf("    * gauss: a single gauss-function at the start.\n");
      printf("    * file <2 filenames>: allows you to specify a file with on "
          "each line a float for both generations.\n");

      return EXIT_FAILURE;
    }
    
    i_max = atoi(argv[1]);
    t_max = atoi(argv[2]);

    if (i_max < 3) {
      printf("argument error: i_max should be >2.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return EXIT_FAILURE;
    }
    if (t_max < 1) {
      printf("argument error: t_max should be >=1.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return EXIT_FAILURE;
    }
    
    iPerTask = i_max / num_tasks;
    
    old = malloc((i_max + 1) * sizeof(double));
    current = malloc((i_max + 1) * sizeof(double));
    next = malloc((i_max + 1) * sizeof(double));
    
    startPtrs = malloc(num_tasks * sizeof(int));
    
    if(old == NULL || current == NULL || next == NULL || startPtrs == NULL){
      fprintf(stderr, "Could not allocate enough memory, aborting.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return EXIT_FAILURE;
    }
    
    memset(old, 0, (i_max + 1) * sizeof(double));
    memset(current, 0, (i_max + 1) * sizeof(double));
    memset(next, 0, (i_max + 1) * sizeof(double));
    
    memset(startPtrs, 0, num_tasks * sizeof(int));
    
    double *startOld = old;
    double *startCur = current;
    
    /* Make space for the left halo of the root task.
     * We will never use this halo, but it is there for simplifying
     * the simulate step.
     */
    old++;
    current++;
    
    /* 
     * How should we will our first two generations? This is determined by the
     * optional further commandline arguments.
     */
    if (argc > 3) {
      if (strcmp(argv[3], "sin") == 0) {
        fill(old, 1, i_max/4, 0, 2*3.14, sin);
        fill(current, 2, i_max/4, 0, 2*3.14, sin);
      } else if (strcmp(argv[3], "sinfull") == 0) {
        fill(old, 1, i_max-2, 0, 10*3.14, sin);
        fill(current, 2, i_max-3, 0, 10*3.14, sin);
      } else if (strcmp(argv[3], "gauss") == 0) {
        fill(old, 1, i_max/4, -3, 3, gauss);
        fill(current, 2, i_max/4, -3, 3, gauss);
      } else if (strcmp(argv[3], "file") == 0) {
        if (argc < 6) {
          printf("No files specified!\n");
          MPI_Abort(MPI_COMM_WORLD, 1);
          return EXIT_FAILURE;
        }
        file_read_double_array(argv[4], old, i_max);
        file_read_double_array(argv[5], current, i_max);
      } else {
        printf("Unknown initial mode: %s.\n", argv[3]);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return EXIT_FAILURE;
      }
    } else {
      /* Default to sinus. */
      fill(old, 1, i_max/4, 0, 2*3.14, sin);
      fill(current, 2, i_max/4, 0, 2*3.14, sin);
    }
    
    for(t = 1; t < num_tasks; t++){
      low = t * iPerTask;
      high = (t + 1) * iPerTask - 1;
      if(t + 1 == num_tasks){
        high = i_max - 1;
      }
      nitems = high - low + 1;
      
      MPI_Send(&nitems, 1, MPI_INT, t, 6, MPI_COMM_WORLD);
      MPI_Send(&t_max, 1, MPI_INT, t, 7, MPI_COMM_WORLD);
      
      old = startCur + low + 1;
      current = startCur + low + 1;
      
      startPtrs[t] = low;
      
      MPI_Send(old, nitems, MPI_DOUBLE, t, 8, MPI_COMM_WORLD);
      MPI_Send(current, nitems, MPI_DOUBLE, t, 9, MPI_COMM_WORLD);
    }
    
    // Reset to original pointers.
    old = startOld;
    current = startCur;
    
    
    nitems = iPerTask;
    
    timer_start();
    
    ret = simulate(nitems, t_max, old, current, next, my_rank, num_tasks);
    ret++;
    double *startRet = ret;
        
    // Receive all data back
    for(t = 1; t < num_tasks; t++){
      ret = startRet + startPtrs[t];
      MPI_Recv(ret, nitems + num_tasks, MPI_DOUBLE, t, 5,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    ret = startRet;
    
    time = timer_end();
    
    printf("Took %g seconds\n", time);
    printf("Normalized: %g seconds\n", time / (1. * i_max * t_max));
    
    file_write_double_array("result.txt", ret, i_max);
    
    old = startOld;
    current = startCur;
  }else{
    MPI_Status *status = MPI_STATUS_IGNORE;
    
    MPI_Recv(&nitems, 1, MPI_INT, 0, 6, MPI_COMM_WORLD, status);
    MPI_Recv(&t_max, 1, MPI_INT, 0, 7, MPI_COMM_WORLD, status);
    
    old = malloc((nitems + 2) * sizeof(double));
    current = malloc((nitems + 2) * sizeof(double));
    next = malloc((nitems + 2) * sizeof(double));
    
    if(old == NULL || current == NULL || next == NULL){
      fprintf(stderr, "Could not allocate enough memory, aborting.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
      return EXIT_FAILURE;
    }

    memset(old, 0, (nitems + 2) * sizeof(double));
    memset(current, 0, (nitems + 2) * sizeof(double));
    memset(next, 0, (nitems + 2) * sizeof(double));
    
    // Start putting data at offset 1, increase pointers for this.
    old++;
    current++;
    
    MPI_Recv(old, nitems, MPI_DOUBLE, 0, 8, MPI_COMM_WORLD, status);
    MPI_Recv(current, nitems, MPI_DOUBLE, 0, 9, MPI_COMM_WORLD, status);
    
    // Reset original pointers.
    old--;
    current--;
    
    ret = simulate(nitems, t_max, old, current, next, my_rank, num_tasks);
    
    // Increase pointer to ignore halo.
    ret++;
    
    // Send back calculated data.
    MPI_Send(ret, nitems, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD);
  }

  free(old);
  free(current);
  free(next);
  MPI_Finalize(); // shutdown MPI

  return EXIT_SUCCESS;
}
