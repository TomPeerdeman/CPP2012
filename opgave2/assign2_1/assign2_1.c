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
		double **startPtrs;
		
		/* Parse commandline args */
		// TODO: fit to 80 char limit
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
		
		startPtrs = malloc(num_tasks * sizeof(double *));
		
		if(old == NULL || current == NULL || next == NULL || startPtrs == NULL){
			fprintf(stderr, "Could not allocate enough memory, aborting.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			return EXIT_FAILURE;
		}
		
		memset(old, 0, (i_max + 1) * sizeof(double));
		memset(current, 0, (i_max + 1) * sizeof(double));
		memset(next, 0, (i_max + 1) * sizeof(double));
		
		memset(startPtrs, 0, num_tasks * sizeof(double *));
		
		double *startOld = old;
		double *startCur = current;
		
		/* Make space for the left halo of the root task.
		 * We will never use this halo, but it is there for simplifying
		 * the simulate step.
		 */
		old++;
		current++;
		
		printf("Ptr compare: %p - %p  %x\n", (void *) old, (void *) startOld, sizeof(double));
		
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
		
		printf("Task 0 gets %d items (0-%d)\n", iPerTask, iPerTask - 1);
		printf("Task %d start %lf-%lf\n", t, startCur[1], startCur[200]);
		startPtrs[0] = current;
		
		for(t = 1; t < num_tasks; t++){
			// TODO: Better distrubution of i's, the last one can get much more
			// i's than the rest
			low = t * iPerTask;
			high = (t + 1) * iPerTask - 1;
			if(t + 1 == num_tasks){
				high = i_max - 1;
			}
			nitems = high - low + 1;
			
			printf("Task %d gets %d items (%d-%d)\n", t, nitems, low, high);
			printf("Task %d start %lf-%lf\n", t, startCur[low+1], startCur[high + 1]);
			
			MPI_Send(&nitems, 1, MPI_INT, t, 6, MPI_COMM_WORLD);
			MPI_Send(&t_max, 1, MPI_INT, t, 7, MPI_COMM_WORLD);
			
			old = startPtrs[0] + low;
			current = startPtrs[0] + low;
			
			startPtrs[t] = current;
			
			MPI_Send(old, nitems, MPI_DOUBLE, t, 8, MPI_COMM_WORLD);
			MPI_Send(current, nitems, MPI_DOUBLE, t, 9, MPI_COMM_WORLD);
		}
		
		// Reset to original pointers.
		old = startOld;
		current = startCur;
		
		
		nitems = iPerTask;
		
		timer_start();
		
		printf("Root prepre ptr %p\n", (void *) current);
		
		current = simulate(nitems, t_max, old, current, next, my_rank, num_tasks);
		
		printf("Root node done\n");
		
		printf("Root pre ptr %p\n", (void *) current);
		int i;
		if(my_rank == 0){
			for(i = 0; i < 10; i++){
				printf("Root-pre [%d] %lf\n", i, current[i]);
			}
		}
		
		// Receive all data back
		for(t = 1; t < num_tasks; t++){
			current = startPtrs[t];
			MPI_Recv(current, nitems + num_tasks, MPI_DOUBLE, t, 5,
				MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		
		current = startCur;
		printf("Root post ptr %p\n", (void *) current);
		
		if(my_rank == 0){
			for(i = 0; i < 10; i++){
				printf("Root-post [%d] %lf\n", i, current[i]);
			}
		}
		
		time = timer_end();
		
		printf("Took %g seconds\n", time);
		printf("Normalized: %g seconds\n", time / (1. * i_max * t_max));

		current = startPtrs[0];
		
		if(my_rank == 0){
			for(i = 0; i < 10; i++){
				printf("Root-postpost [%d] %lf\n", i, current[i]);
			}
		}
		
		printf("Root low: %lf\n", current[0]);
		
		file_write_double_array("result.txt", current, i_max);
		
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
		
		printf("Node %d of %d done\n", my_rank, num_tasks);
		
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
