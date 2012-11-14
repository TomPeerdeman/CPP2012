// TODO: remove printf's
#include <stdio.h>

#include "mpi.h"

int broadcast_counter = 0;

int MYMPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, 
	MPI_Comm communicator){
				
    int num_tasks, my_rank, tag;

    // determine total number of tasks, and my mpi process rank
    MPI_Comm_size(communicator, &num_tasks);
    MPI_Comm_rank(communicator, &my_rank);

    // only send if you are the root
    if(my_rank == root){
		broadcast_counter++;
		tag = broadcast_counter;
		printf("Send tag %d from %d\n", tag, my_rank);
		
      // ring structure, last rank connects to first rank and vice versa
      if(root != num_tasks-1)
        MPI_Send(buffer, count, datatype, root+1, tag, communicator);
      else
        MPI_Send(buffer, count, datatype, 0, tag, communicator);
      if(root != 0)
        MPI_Send(buffer, count, datatype, root-1, tag, communicator);
      else
        MPI_Send(buffer, count, datatype, num_tasks-1, tag, communicator);
    }
    
    // receive if you are not the root
    else{
		MPI_Status status;
		MPI_Recv(buffer, count, datatype, MPI_ANY_SOURCE, MPI_ANY_TAG,
			communicator, &status);
		int sender = status.MPI_SOURCE;
		tag = status.MPI_TAG;
		printf("%d received tag %d from %d\n", my_rank, tag, sender);
		int send_to;
		
		if(tag > broadcast_counter){
			broadcast_counter = tag;
			
			if(sender < my_rank){
				send_to = (sender + 2) % num_tasks;
			}else{
				send_to = (sender - 2 + num_tasks) % num_tasks;
			}
			printf("%d resend tag %d to %d\n", my_rank, tag, send_to);
			MPI_Send(buffer, count, datatype, send_to, tag, communicator);
		}
    }
	
	return 0;
}
