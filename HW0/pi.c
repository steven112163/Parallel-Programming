#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define NUMBER_OF_TOSSES 1000000


int main(void) {
    srand(time(NULL));

    long long number_in_circle = 0;
    for (long long toss = 0; toss < NUMBER_OF_TOSSES; toss++) {
        double x = rand() / ((double) RAND_MAX) * 2 - 1;
        double y = rand() / ((double) RAND_MAX) * 2 - 1;
        if (x * x + y * y <= 1)
            number_in_circle++;
    }

    printf("%.2f\n", 4 * number_in_circle / ((double) NUMBER_OF_TOSSES));

    return 0;
}