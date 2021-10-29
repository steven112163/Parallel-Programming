#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <pthread.h>
#include "SIMDxorshift/include/simdxorshift128plus.h"
#include <immintrin.h>

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

    // Setup SIMD random number generator
    unsigned int seed = (unsigned int) time(NULL);
    avx_xorshift128plus_key_t key;
    uint64_t key_1 = rand_r(&seed);
    uint64_t key_2 = rand_r(&seed);
    if (key_1 == 0)
        key_1 = 1;
    if (key_2 == 0)
        key_2 = 1;
    avx_xorshift128plus_init(key_1, key_2, &key);

    __m256 max = _mm256_set1_ps((float) RAND_MAX);
    __m256 ones = _mm256_set1_ps((float) 1);

    // Start tossing
    for (long long toss = 0; toss < num_of_tosses; toss += 8) {
        // Get x^2
        __m256i int_x = avx_xorshift128plus(&key);
        __m256 float_x = _mm256_cvtepi32_ps(int_x);
        __m256 x = _mm256_div_ps(float_x, max);
        __m256 x_squared = _mm256_mul_ps(x, x);

        // Get y^2
        __m256i int_y = avx_xorshift128plus(&key);
        __m256 float_y = _mm256_cvtepi32_ps(int_y);
        __m256 y = _mm256_div_ps(float_y, max);
        __m256 y_squared = _mm256_mul_ps(y, y);

        // Calculate whether x^2 + y^2 <= 1
        __m256 distance = _mm256_add_ps(x_squared, y_squared);
        __m256 result = _mm256_cmp_ps(distance, ones, _CMP_LE_OQ);

        float val[8];
        _mm256_store_ps(val, result);

        // If x^2 + y^2 > 1, then the value is 0
        for (int idx = 0; idx < 8; idx++)
            if (val[idx] != 0)
                num_in_circle++;
    }

    return (void *) num_in_circle;
}