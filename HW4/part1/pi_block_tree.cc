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
    MPI_Status status;
    int tag = 0;

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

    // TODO: binary tree reduction
    long long received_num_in_circle;
    for (int step = 2; step <= world_size; step *= 2) {
        if (world_rank % step == 0) {
            int source = world_rank + (int) (step / 2);
            MPI_Recv(&received_num_in_circle, 1, MPI_LONG, source, tag, MPI_COMM_WORLD, &status);
            num_in_circle += received_num_in_circle;
        } else {
            int target = world_rank - (int) (step / 2);
            MPI_Send(&num_in_circle, 1, MPI_LONG, target, tag, MPI_COMM_WORLD);
            break;
        }
    }

    if (world_rank == 0) {
        // TODO: PI result
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
