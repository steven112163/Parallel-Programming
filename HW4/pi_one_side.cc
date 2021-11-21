#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv) {
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: MPI init
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Win win;

    // Start tossing
    unsigned int seed = (unsigned int) (world_rank + 1) * time(NULL);

    double max_div_2 = (double) RAND_MAX / 2;
    long long num_of_tosses = (long long) tosses / world_size;
    long long num_in_circle = 0;

    for (long long toss = 0; toss < num_of_tosses; toss++) {
        double x = (double) rand_r(&seed) / max_div_2 - 1;
        double y = (double) rand_r(&seed) / max_div_2 - 1;
        if (x * x + y * y <= 1)
            num_in_circle++;
    }

    if (world_rank == 0) {
        // Master
        long long *buffer;
        MPI_Alloc_mem((world_size - 1) * sizeof(long long), MPI_INFO_NULL, &buffer);
        MPI_Win_create(buffer,
                       (world_size - 1) * sizeof(long long),
                       sizeof(long long),
                       MPI_INFO_NULL,
                       MPI_COMM_WORLD,
                       &win);

        MPI_Barrier(MPI_COMM_WORLD);
        for (int source = 1; source < world_size; source++)
            num_in_circle += buffer[source - 1];

        MPI_Free_mem(buffer);
    } else {
        // Workers
        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
        MPI_Put(&num_in_circle, 1, MPI_LONG, 0, world_rank - 1, 1, MPI_LONG, win);
        MPI_Win_unlock(0, win);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Win_free(&win);

    if (world_rank == 0) {
        // TODO: handle PI result
        pi_result = 4.0 * num_in_circle / ((double) tosses);

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}