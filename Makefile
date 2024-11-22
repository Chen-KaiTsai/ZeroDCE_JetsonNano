main: main.o qKernel.o qBasic.o qAlgorithm.o qAlgorithm_ZeroCopy.o
	@nvcc -o main *.o -O3 -m64 -I/usr/local/cuda/samples/common/inc -gencode arch=compute_53,code=sm_53 --compiler-options -Wall `pkg-config opencv4 --cflags --libs` -Xcompiler -fopenmp

qKernel.o: qKernel.cu
	@nvcc -c -O3 qKernel.cu -I/usr/local/cuda/samples/common/inc `pkg-config opencv4 --cflags --libs` -gencode arch=compute_53,code=sm_53 -Xcompiler -fopenmp

qBasic.o: qBasic.cpp
	@nvcc -c -O3 qBasic.cpp -I/usr/local/cuda/samples/common/inc `pkg-config opencv4 --cflags --libs` -gencode arch=compute_53,code=sm_53 -Xcompiler -fopenmp

qAlgorithm.o: qAlgorithm.cu qBasic.o qKernel.o
	@nvcc -c -O3 qAlgorithm.cu -I/usr/local/cuda/samples/common/inc `pkg-config opencv4 --cflags --libs` -gencode arch=compute_53,code=sm_53 -Xcompiler -fopenmp

qAlgorithm_ZeroCopy.o: qAlgorithm_ZeroCopy.cu qBasic.o qKernel.o
	@nvcc -c -O3 qAlgorithm_ZeroCopy.cu -I/usr/local/cuda/samples/common/inc `pkg-config opencv4 --cflags --libs` -gencode arch=compute_53,code=sm_53 -Xcompiler -fopenmp

main.o: main.cpp
	@nvcc -c -O3 main.cpp -I/usr/local/cuda/samples/common/inc `pkg-config opencv4 --cflags --libs` -gencode arch=compute_53,code=sm_53 -Xcompiler -fopenmp