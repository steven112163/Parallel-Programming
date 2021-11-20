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

    // TODO: use MPI_Gather
    long long *buffer;
    if (world_rank == 0)
        buffer = (long long *) malloc(world_size * sizeof(long long));
    MPI_Gather(&num_in_circle, 1, MPI_LONG, buffer, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // TODO: PI result
        for (int idx = 1; idx < world_size; idx++)
            num_in_circle += buffer[idx];
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
