#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

#define NUMBER_OF_TOSSES 1000000


int main(void) {
    std::srand(std::time(NULL));

    long long number_in_circle = 0;
    for (long long toss = 0; toss < NUMBER_OF_TOSSES; toss++) {
        double x = std::rand() / ((double) RAND_MAX) * 2 - 1;
        double y = std::rand() / ((double) RAND_MAX) * 2 - 1;
        if (x * x + y * y <= 1)
            number_in_circle++;
    }
    std::cout << std::setprecision(3) << 4 * number_in_circle / ((double) NUMBER_OF_TOSSES) << std::endl;

    return 0;
}