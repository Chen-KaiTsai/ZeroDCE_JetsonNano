main: main.o qAlgorithm.o
	@nvcc -o main *.o -g -m64 -I/usr/local/cuda/samples/common/inc -gencode arch=compute_53,code=sm_53 --compiler-options -Wall `pkg-config opencv4 --cflags --libs` -Xcompiler -fopenmp

qAlgorithm.o:
	@nvcc -c -g qAlgorithm.cu -I/usr/local/cuda/samples/common/inc `pkg-config opencv4 --cflags --libs` -gencode arch=compute_53,code=sm_53 -Xcompiler -fopenmp

main.o:
	@nvcc -c -g main.cpp -I/usr/local/cuda/samples/common/inc `pkg-config opencv4 --cflags --libs` -gencode arch=compute_53,code=sm_53 -Xcompiler -fopenmp
