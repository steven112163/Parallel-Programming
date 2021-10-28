#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>

void *generate_tosses(void *tosses_in_thread);

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: ./pi.out <num_of_threads> <num_of_tosses>");
        exit(1);
    }

    // Get arguments
    long num_of_threads = strtol(argv[1], NULL, 10);
    long long total_tosses = strtoll(argv[2], NULL, 10);

    // Calculate the number of tosses that each thread has to deal with
    long long tosses_in_thread = (long long) (total_tosses / num_of_threads);
    long long remaining_tosses = total_tosses - tosses_in_thread * num_of_threads;

    // Create threads and initialize random states
    pthread_t *threads = malloc(num_of_threads * sizeof(pthread_t));
    for (long idx = 0; idx < num_of_threads; idx++) {
        if (idx == 0)
            pthread_create(&threads[idx], NULL, generate_tosses, (void *) (tosses_in_thread + remaining_tosses));
        else
            pthread_create(&threads[idx], NULL, generate_tosses, (void *) tosses_in_thread);
    }

    // Aggregate results
    long long num_in_circle = 0;
    void *return_value;
    for (long idx = 0; idx < num_of_threads; idx++) {
        pthread_join(threads[idx], &return_value);
        num_in_circle += (long long) return_value;
    }

    printf("%.6f\n", 4.0 * num_in_circle / ((double) total_tosses));

    // Free space
    free(threads);

    return 0;
}

void *generate_tosses(void *tosses_in_thread) {
    long long num_of_tosses = (long long) tosses_in_thread;
    long long num_in_circle = 0;

    unsigned int seed = (unsigned int) time(NULL);
    double max_div_2 = (double) RAND_MAX / 2;

    for (long long toss = 0; toss < num_of_tosses; toss++) {
        double x = (double) rand_r(&seed) / max_div_2 - 1;
        double y = (double) rand_r(&seed) / max_div_2 - 1;
        if (x * x + y * y <= 1)
            num_in_circle++;
    }

    return (void *) num_in_circle;
}