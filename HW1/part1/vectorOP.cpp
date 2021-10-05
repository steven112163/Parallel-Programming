#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N) {
    __pp_vec_float x;
    __pp_vec_float result;
    __pp_vec_float zero = _pp_vset_float(0.f);
    __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i = 0; i < N; i += VECTOR_WIDTH) {

        // All ones
        maskAll = _pp_init_ones();

        // All zeros
        maskIsNegative = _pp_init_ones(0);

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll); // x = values[i];

        // Set mask according to predicate
        _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

        // Execute instruction using mask ("if" clause)
        _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

        // Inverse maskIsNegative to generate "else" mask
        maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

        // Execute instruction ("else" clause)
        _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

        // Write results back to memory
        _pp_vstore_float(output + i, result, maskAll);
    }
}

void clampedExpVector(float *values, int *exponents, float *output, int N) {
    //
    // PP STUDENTS TODO: Implement your vectorized version of
    // clampedExpSerial() here.
    //
    // Your solution should work for any value of
    // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
    //
    __pp_vec_float x, result;
    __pp_vec_int y, count;
    __pp_vec_int zero = _pp_vset_int(0);
    __pp_vec_int int_one = _pp_vset_int(1);
    __pp_vec_float float_one = _pp_vset_float(1.f);
    __pp_vec_float nine = _pp_vset_float(9.999999f);

    __pp_mask maskAll = _pp_init_ones();
    __pp_mask maskIsZero, maskIsNotZero, maskGtNine, maskPart;

    int i;
    for (i = 0; i < N; i += VECTOR_WIDTH) {
        // Handle (N % VECTOR_WIDTH) != 0
        if (i + VECTOR_WIDTH > N)
            break;

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll); // x = values[i];
        // Load vector of exponents from contiguous memory addresses
        _pp_vload_int(y, exponents + i, maskAll); // y = exponents[i];

        // Set mask according to predicate
        _pp_veq_int(maskIsZero, y, zero, maskAll); // if (y == 0) {

        // Execute instruction ("if" clause)
        _pp_vmove_float(result, float_one, maskIsZero); // output[i] = 1.f;

        // Inverse maskIsZero to generate "else" mask
        maskIsNotZero = _pp_mask_not(maskIsZero); // } else {

        // Execute instruction ("else" clause)
        // Move x to result
        _pp_vmove_float(result, x, maskIsNotZero); // result = x;
        // Get count
        _pp_vsub_int(count, y, int_one, maskIsNotZero); // count = y - 1;
        // Get new non-zero mask
        _pp_vgt_int(maskIsNotZero, count, zero, maskIsNotZero);
        // While loop
        while (_pp_cntbits(maskIsNotZero) > 0) { // while (count > 0) {
            // Multiply result by x
            _pp_vmult_float(result, result, x, maskIsNotZero); // result *= x;
            // Subtract count by 1
            _pp_vsub_int(count, count, int_one, maskIsNotZero); // count--;
            // Get new non-zero mask
            _pp_vgt_int(maskIsNotZero, count, zero, maskIsNotZero);
        }
        // Set mask according to predicate
        _pp_vgt_float(maskGtNine, result, nine, maskAll); // if (result > 9.999999f) {
        // Execute instruction ("if" clause)
        _pp_vmove_float(result, nine, maskGtNine); // result = 9.999999f;

        // Write results back to memory
        _pp_vstore_float(output + i, result, maskAll); // output[i] = result;
    }

    // Handle (N % VECTOR_WIDTH) != 0
    if (i + VECTOR_WIDTH > N) {
        maskPart = _pp_init_ones(N - i);
        maskIsZero = _pp_init_ones(0);

        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskPart); // x = values[i];
        // Load vector of exponents from contiguous memory addresses
        _pp_vload_int(y, exponents + i, maskPart); // y = exponents[i];

        // Set mask according to predicate
        _pp_veq_int(maskIsZero, y, zero, maskPart); // if (y == 0) {

        // Execute instruction ("if" clause)
        _pp_vmove_float(result, float_one, maskIsZero); // output[i] = 1.f;

        // Inverse maskIsZero to generate "else" mask
        maskIsNotZero = _pp_mask_not(maskIsZero); // } else {
        maskIsNotZero = _pp_mask_and(maskIsNotZero, maskPart);

        // Execute instruction ("else" clause)
        // Move x to result
        _pp_vmove_float(result, x, maskIsNotZero); // result = x;
        // Get count
        _pp_vsub_int(count, y, int_one, maskIsNotZero); // count = y - 1;
        // Get new non-zero mask
        _pp_vgt_int(maskIsNotZero, count, zero, maskIsNotZero);
        // While loop
        while (_pp_cntbits(maskIsNotZero) > 0) { // while (count > 0) {
            // Multiply result by x
            _pp_vmult_float(result, result, x, maskIsNotZero); // result *= x;
            // Subtract count by 1
            _pp_vsub_int(count, count, int_one, maskIsNotZero); // count--;
            // Get new non-zero mask
            _pp_vgt_int(maskIsNotZero, count, zero, maskIsNotZero);
        }
        // Set mask according to predicate
        _pp_vgt_float(maskGtNine, result, nine, maskPart); // if (result > 9.999999f) {
        // Execute instruction ("if" clause)
        _pp_vmove_float(result, nine, maskGtNine); // result = 9.999999f;

        // Write results back to memory
        _pp_vstore_float(output + i, result, maskPart); // output[i] = result;
    }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N) {
    //
    // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
    //

    float result[VECTOR_WIDTH];
    bool power_of_2 = ((int) VECTOR_WIDTH / 2) % 2 == 0;
    float sum = 0.0;

    int number_of_hadd_and_interleave = 0;
    if (power_of_2) {
        int current = VECTOR_WIDTH;
        while (current > 1) {
            number_of_hadd_and_interleave++;
            current /= 2;
        }
    } else
        number_of_hadd_and_interleave = 1;

    __pp_vec_float x, y;
    __pp_mask maskAll = _pp_init_ones();
    for (int i = 0; i < N; i += VECTOR_WIDTH) {
        // Load vector of values from contiguous memory addresses
        _pp_vload_float(x, values + i, maskAll);

        for (int j = 0; j < number_of_hadd_and_interleave; j++) {
            // Interleave
            _pp_interleave_float(y, x);

            // Move result back to x
            _pp_vmove_float(x, y, maskAll);

            // Add adjacent elements
            _pp_hadd_float(x, x);
        }

        if (!power_of_2) {
            // Interleave
            _pp_interleave_float(y, x);

            // Move result back to x
            _pp_vmove_float(x, y, maskAll);
        }

        // Store vector to result
        _pp_vstore_float(result, x, maskAll);

        if (power_of_2)
            sum += result[0];
        else {
            for (int j = 0; j < VECTOR_WIDTH / 2; j++)
                sum += result[j];
        }
    }

    return sum;
}