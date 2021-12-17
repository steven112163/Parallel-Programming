default: conv

CC = gcc-10
FLAGS = -O3 -lOpenCL -m64 -ffloat-store -w -g -D CL_TARGET_OPENCL_VERSION=220

NVCC = nvcc
CUDA_LINK_FLAGS = -rdc=true -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC'
CUDA_COMPILE_FLAGS = --device-c -gencode=arch=compute_61,code=sm_61 -Xcompiler '-fPIC' -g -O3

OBJS = main.o bmpfuncs.o serialConv.o helper.o

conv: $(OBJS)
	$(CC) -c $(FLAGS) hostFE.c -o hostFE.o
	$(CC) -o $@ $(OBJS) hostFE.o $(FLAGS)

%.o: %.c
	$(CC) -c $(FLAGS) $< -o $@

cuda: $(OBJS)
	$(NVCC) $(CUDA_COMPILE_FLAGS) -c hostFE.cu -o hostFE.o
	$(NVCC) $(CUDA_LINK_FLAGS) -o conv $(OBJS) hostFE.o -lOpenCL

clean:
	rm -f conv *.o output.bmp ref.bmp
