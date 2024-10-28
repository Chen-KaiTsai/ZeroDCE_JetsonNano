main: main.o qKernel.o qBasic.o qAlgorithm.o qAlgorithm_ZeroCopy.o
	@nvcc -o main *.o -g -m64 -I/usr/local/cuda/samples/Common/ -gencode arch=compute_86,code=sm_86 --compiler-options -Wall `pkg-config opencv4 --cflags --libs` -Xcompiler -fopenmp

qKernel.o: qKernel.cu
	@nvcc -c -g qKernel.cu -I/usr/local/cuda/samples/Common/ `pkg-config opencv4 --cflags --libs` -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp

qBasic.o: qBasic.cpp
	@nvcc -c -g qBasic.cpp -I/usr/local/cuda/samples/Common/ `pkg-config opencv4 --cflags --libs` -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp

qAlgorithm.o: qAlgorithm.cu qBasic.o qKernel.o
	@nvcc -c -g qAlgorithm.cu -I/usr/local/cuda/samples/Common/ `pkg-config opencv4 --cflags --libs` -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp

qAlgorithm_ZeroCopy.o: qAlgorithm_ZeroCopy.cu qBasic.o qKernel.o
	@nvcc -c -g qAlgorithm_ZeroCopy.cu -I/usr/local/cuda/samples/Common/ `pkg-config opencv4 --cflags --libs` -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp

main.o: main.cpp
	@nvcc -c -g main.cpp -I/usr/local/cuda/samples/Common/ `pkg-config opencv4 --cflags --libs` -gencode arch=compute_86,code=sm_86 -Xcompiler -fopenmp
