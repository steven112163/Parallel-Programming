__kernel void convolution(int filterWidth,
                          __constant float *filter,
                          int imageHeight,
                          int imageWidth,
                          __global float *inputImage,
                          __global float *outputImage) {
    int thisX = get_global_id(0);
    int thisY = get_global_id(1);

    int halfFilterSize = filterWidth / 2;
    float sum = 0.0f;
    int row, col;
    for (row = -halfFilterSize; row <= halfFilterSize; row++) {
        for (col = -halfFilterSize; col <= halfFilterSize; col++) {
            if (thisY + row >= 0 && thisY + row < imageHeight &&
                thisX + col >= 0 && thisX + col < imageWidth) {
                sum += inputImage[(thisY + row) * imageWidth + thisX + col] *
                filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize];
            }
        }
    }
    outputImage[thisY * imageWidth + thisX] = sum;
}
