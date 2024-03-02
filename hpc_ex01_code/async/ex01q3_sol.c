#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

void do_work(int i) {
    printf("processing %d\n", i);
    sleep(5);
}

int main(int argc, char** argv) {
    int rank;
    int size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
        printf("Running with %d MPI processes\n", size);

    int M = 2;  // two tasks per process
    int N = M * size;
    int input[N]; // input table that stores all data to be sent
    int received_input;

    if (rank == 0) {
        srand48(time(0));

        for (int i = 0; i < N; i++) {
            input[i] = lrand48() % 1000;  // some random value
        }
    }

    // Scatter for comunication
    int receive_buf[M];
    MPI_Scatter(input, M, MPI_INT, receive_buf, M, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < M; i++) {
        do_work(receive_buf[i]); 
    }

    MPI_Finalize();
    return 0;
}
