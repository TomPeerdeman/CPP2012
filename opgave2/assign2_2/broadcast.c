#include "mpi.h"

int MYMPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, 
	MPI_Comm communicator){
				
    int num_tasks, my_rank;

    MPI_Comm_size(communicator, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if(my_rank == root){
      if(root != num_tasks-1)
        MPI_Send(buffer, count, datatype, root+1, 0, communicator);
      else
        MPI_Send(buffer, count, datatype, 0, 0, communicator);
      if(root != 0)
        MPI_Send(buffer, count, datatype, root-1, 0, communicator);
      else
        MPI_Send(buffer, count, datatype, num_tasks-1, 0, communicator);
    }
    else{
      MPI_Recv(buffer, count, datatype, root, 0, communicator, MPI_STATUS_IGNORE);
    }
		
		
		
	return 0;
}
