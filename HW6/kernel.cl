__kernel void convolution(int filterWidth,
                          __constant float *filter,
                          __read_only __global float *inputImage,
                          __write_only __global float *outputImage) {
    int thisX = get_global_id(0);
    int thisY = get_global_id(1);
    int imageWidth = get_global_size(0);
    int imageHeight = get_global_size(1);
    int halfFilterSize = filterWidth >> 1;

    float sum = 0;
    int row, col;
    for (row = -halfFilterSize; row <= halfFilterSize; row++) {
        for (col = -halfFilterSize; col <= halfFilterSize; col++) {
            if (filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize] != 0 &&
                thisY + row >= 0 && thisY + row < imageHeight &&
                thisX + col >= 0 && thisX + col < imageWidth) {
                sum += inputImage[(thisY + row) * imageWidth + thisX + col] *
                filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize];
            }
        }
    }
    outputImage[thisY * imageWidth + thisX] = sum;
}
