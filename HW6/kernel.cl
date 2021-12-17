__kernel void convolution(int filterWidth,
                          __constant float *filter,
                          int imageHeight,
                          int imageWidth,
                          __global float *inputImage,
                          __global float *outputImage) {
    const int thisX = get_global_id(0);
    const int thisY = get_global_id(1);

    int halfFilterSize = filterWidth / 2;
    float sum = 0.0f;
    int row, col;
    for (row = -halfFilterSize; row < halfFilterSize + 1; row++) {
        for (col = -halfFilterSize; col < halfFilterSize + 1; col++) {
            if (thisY + row > -1 && thisY + row < imageHeight &&
                thisX + col > -1 && thisX + col < imageWidth) {
                sum += inputImage[(thisY + row) * imageWidth + thisX + col] *
                filter[(row + halfFilterSize) * filterWidth + col + halfFilterSize];
            }
        }
    }
    outputImage[thisY * imageWidth + thisX] = sum;
}
