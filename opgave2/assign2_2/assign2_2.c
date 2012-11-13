#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#include "broadcast.h"

int main(int argc, char **argv){
	int rc, my_rank, num_tasks;

	rc = MPI_Init(&argc, &argv);
	if(rc != MPI_SUCCESS){
		fprintf(stderr, "Unable to set up MPI");
		// Abort MPI runtime.
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	// Broadcast here
	
	MPI_Finalize(); // shutdown MPI
}