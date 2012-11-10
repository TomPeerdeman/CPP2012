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
double gauss(double x)
{
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
        double sample_end, func_t f)
{
    int i;
    float dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++) {
        array[i + offset] = f(sample_start + i * dx);
    }
}


int main(int argc, char *argv[])
{
    double *old, *current, *next, *ret;
    int t_max, i_max;
    double time;
    int rc , num_tasks , my_rank;
    int iPerTask;

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
        return EXIT_FAILURE;
    }
    if (t_max < 1) {
        printf("argument error: t_max should be >=1.\n");
        return EXIT_FAILURE;
    }
    
    //Setting up MPI
    rc = MPI_Init (&argc, &argv);
	if ( rc != MPI_SUCCESS ) { // Check for success
        fprintf ( stderr , " Unable to set up MPI ");
        MPI_Abort ( MPI_COMM_WORLD , rc ); // Abort MPI runtime
    }
    MPI_Comm_size ( MPI_COMM_WORLD , &num_tasks ); // Determine number of tasks
    MPI_Comm_rank ( MPI_COMM_WORLD , &my_rank );

    //amount of i each task should run
    iPerTask = i_max/num_tasks;
    
    /* Allocate and initialize buffers, with only the size of i's per task
     * that should be calculated. If a task is at the beginning or the end,
     * only one halo cell is needed, else 2 halo cells are needed.
     */
    if(my_rank == 0 || my_rank == num_tasks-1){
      old = malloc((iPerTask+1) * sizeof(double));
      current = malloc((iPerTask+1) * sizeof(double));
      next = malloc((iPerTask+1) * sizeof(double));
    }
    
    else{
      old = malloc((iPerTask+2) * sizeof(double));
      current = malloc((iPerTask+2) * sizeof(double));
      next = malloc((iPerTask+2) * sizeof(double));
    }

    if (old == NULL || current == NULL || next == NULL) {
        fprintf(stderr, "Could not allocate enough memory, aborting.\n");
        return EXIT_FAILURE;
    }

    if(my_rank == 0 || my_rank == num_tasks-1){
      memset(old, 0, (iPerTask+1) * sizeof(double));
      memset(current, 0, (iPerTask+1) * sizeof(double));
      memset(next, 0, (iPerTask+1) * sizeof(double));
    }
    else{
      memset(old, 0, (iPerTask+2) * sizeof(double));
      memset(current, 0, (iPerTask+2) * sizeof(double));
      memset(next, 0, (iPerTask+2) * sizeof(double));
    }

    /* How should we will our first two generations? This is determined by the
     * optional further commandline arguments.
     * */
     //TODO? change i_max here to iPerTask?
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
                return EXIT_FAILURE;
            }
            file_read_double_array(argv[4], old, i_max);
            file_read_double_array(argv[5], current, i_max);
        } else {
            printf("Unknown initial mode: %s.\n", argv[3]);
            return EXIT_FAILURE;
        }
    } else {
        /* Default to sinus. */
        fill(old, 1, i_max/4, 0, 2*3.14, sin);
        fill(current, 2, i_max/4, 0, 2*3.14, sin);
    }

    timer_start();

    /* Call the actual simulation that should be implemented in simulate.c. */
    ret = simulate(i_max, t_max, old, current, next, my_rank, num_tasks);

    time = timer_end();
    printf("Took %g seconds\n", time);
    printf("Normalized: %g seconds\n", time / (1. * i_max * t_max));

    file_write_double_array("result.txt", ret, i_max);

    free(old);
    free(current);
    free(next);
    MPI_Finalize(); // shutdown MPI

    return EXIT_SUCCESS;
}
