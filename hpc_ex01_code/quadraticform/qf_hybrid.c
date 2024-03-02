#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <sys/time.h>

double get_wtime(void){
	struct timeval t;
	gettimeofday(&t, NULL);
	return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}

int main(int argc, char** argv){
	int n = 16384;

	if (argc == 2)
		n = atoi(argv[1]);


	double *A = (double *)malloc(n*n*sizeof(double));
	double *v = (double *)malloc(n*sizeof(double));
	double *w = (double *)malloc(n*sizeof(double));
	double result = 0.0;
	int rank;
	int size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_local = n/size;
     if (rank == size - 1) {
        n_local += n % size;
    }

    int local_start = rank*n_local;
    int local_end;
    if(rank == size - 1) {
        local_end = n;
    }else {
        local_end = local_start + n_local;
    }

	double t0 = get_wtime();

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            int i, j;
            #pragma omp parallel for shared(A,n) private(i,j)
            for (i=local_start; i<local_end; i++)
                for (j=0; j<n; j++)
                    A[i*n+j] = (i + 2.0*j) / (n*n);
        }

        #pragma omp section
        {
            int i;
            #pragma omp parallel for shared(v,n) private(i)
            for (i=local_start; i<local_end; i++)
                v[i] = 1.0 + 2.0 / (i + 0.5);
        }

        #pragma omp section
        {
            int i;
            #pragma omp parallel for shared(w,n) private(i)
            for (i=local_start; i<local_end; i++)
                w[i] = 1.0 - i / (3.0*n);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD); 

    int sendcounts[size];
    int displs[size];
    for(int i = 0; i < size - 1; i++) {
        sendcounts[i] = n_local;
        displs[i] = i * n_local;
    }

    sendcounts[size - 1] = n_local + (n % size);
    displs[size - 1] = (size - 1) * n_local;

    if (rank == 0) {
        MPI_Gatherv(MPI_IN_PLACE, n_local, MPI_DOUBLE, v, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gatherv(&v[local_start], n_local, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){  
        MPI_Gatherv(MPI_IN_PLACE, n_local, MPI_DOUBLE, w, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gatherv(&w[local_start], n_local, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if(rank == 0){
        for(int i = 0; i < size; i++) {
            sendcounts[i] *= n;
            displs[i] *= n;
        }

        MPI_Gatherv(MPI_IN_PLACE, n_local*n, MPI_DOUBLE, A, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }else{
        MPI_Gatherv(&A[local_start*n], n_local*n, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        int i, j;
        #pragma omp parallel for shared(v,w,A,n) private(i, j) reduction(+:result) collapse(2)
        for (i=0; i<n; i++){
            for (j=0; j<n; j++)
                result += v[i] * A[i*n + j] * w[j];
        }

        double t1 = get_wtime();
        printf("Result = %lf\n", result);
        printf("Elapsed time: %lf sec.\n", t1 - t0);
    }

	MPI_Finalize();

	free(A);
	free(v);
	free(w);

	return 0;
}