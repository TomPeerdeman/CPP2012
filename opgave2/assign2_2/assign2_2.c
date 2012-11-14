#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#include "broadcast.h"

int main(int argc, char **argv){
	int rc, my_rank, num_tasks, rootnode;

	rc = MPI_Init(&argc, &argv);
	if(rc != MPI_SUCCESS){
		fprintf(stderr, "Unable to set up MPI\n");
		// Abort MPI runtime.
		MPI_Abort(MPI_COMM_WORLD, rc);
	}
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	
	if(argc < 2){
		printf("No root node given.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	if(argc < 3){
		printf("No message given.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	rootnode = atoi(argv[1]);
	if(rootnode < 0 || rootnode > num_tasks -1){
		printf("Invalid root node.\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}
	
	if(my_rank != rootnode){
		char *ptr = malloc(100 * sizeof(char));
		// Recieve broadcast
		MYMPI_Bcast(ptr, 100, MPI_CHAR, rootnode, MPI_COMM_WORLD);
		printf("%d received: %s\n", my_rank, ptr);
		free(ptr);
	}else{
		// Send broadcast
		printf("%d sending %s\n", my_rank, argv[2]);
		MYMPI_Bcast(argv[2], strlen(argv[2]) + 1, MPI_CHAR, rootnode,
			MPI_COMM_WORLD);
	}
	
	MPI_Finalize(); // shutdown MPI
}