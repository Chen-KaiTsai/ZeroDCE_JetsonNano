#include "main.cuh"
#include "qWeight.h"

RGBIOData_t* INDATA = nullptr;
RGBIOData_t* dINDATA = nullptr;
RGBIOData_t* OUTDATA = nullptr;
RGBIOData_t* dOUTDATA = nullptr;
qNormImg_t* dNORM = nullptr;
qNetIO_t* NETIO = nullptr;
qNetIO_t* dNETIO = nullptr;
qNetFeature_t* dFEATURE1 = nullptr;
qNetFeature_t* dFEATURE2 = nullptr;
qEnhancedParam_t* PARAM = nullptr;
qEnhancedParam_t* dPARAM = nullptr;
qEnhancedParam_t* UPSBUFFER = nullptr;
qWConv1st_t* CONVW01 = nullptr;
qBConv1st_t* CONVB01 = nullptr;
qWConv2nd_t* CONVW02 = nullptr;
qBConv2nd_t* CONVB02 = nullptr;
qWConv3rd_t* CONVW03 = nullptr;
qBConv3rd_t* CONVB03 = nullptr;
qWConv1st_t* dCONVW01 = nullptr;
qBConv1st_t* dCONVB01 = nullptr;
qWConv2nd_t* dCONVW02 = nullptr;
qBConv2nd_t* dCONVB02 = nullptr;
qWConv3rd_t* dCONVW03 = nullptr;
qBConv3rd_t* dCONVB03 = nullptr;

void DCE::initMem() {
    INDATA = (RGBIOData*)malloc(sizeof(RGBIOData_t));
    
    OUTDATA = (RGBIOData*)malloc(sizeof(RGBIOData_t));

    NETIO = (qNetIO_t*)malloc(sizeof(qNetIO_t));
    UPSBUFFER = (qEnhancedParam_t*)malloc(sizeof(qEnhancedParam_t));
    PARAM = (qEnhancedParam_t*)malloc(sizeof(qEnhancedParam_t));
    
    CONVW01 = (qWConv1st_t*)malloc(sizeof(qWConv1st_t));
    CONVB01 = (qBConv1st_t*)malloc(sizeof(qBConv1st_t));
    CONVW02 = (qWConv2nd_t*)malloc(sizeof(qWConv2nd_t));
    CONVB02 = (qBConv2nd_t*)malloc(sizeof(qBConv2nd_t));
    CONVW03 = (qWConv3rd_t*)malloc(sizeof(qWConv3rd_t));
    CONVB03 = (qBConv3rd_t*)malloc(sizeof(qBConv3rd_t));
}

void DCE::initMem_ZeroCopy() {
    cudaHostAlloc((void**)&INDATA, sizeof(RGBIOData_t), cudaHostAllocMapped);

    cudaHostAlloc((void**)&OUTDATA, sizeof(RGBIOData_t), cudaHostAllocMapped);

    cudaHostAlloc((void**)&NETIO, sizeof(qNetIO_t), cudaHostAllocMapped);
    cudaHostAlloc((void**)&UPSBUFFER, sizeof(qEnhancedParam_t), cudaHostAllocMapped);
    cudaHostAlloc((void**)&PARAM, sizeof(qEnhancedParam_t), cudaHostAllocMapped);
    
    cudaHostAlloc((void**)&CONVW01, sizeof(qWConv1st_t), cudaHostAllocMapped);
    cudaHostAlloc((void**)&CONVB01, sizeof(qBConv1st_t), cudaHostAllocMapped);
    cudaHostAlloc((void**)&CONVW02, sizeof(qWConv2nd_t), cudaHostAllocMapped);
    cudaHostAlloc((void**)&CONVB02, sizeof(qBConv2nd_t), cudaHostAllocMapped);
    cudaHostAlloc((void**)&CONVW03, sizeof(qWConv3rd_t), cudaHostAllocMapped);
    cudaHostAlloc((void**)&CONVB03, sizeof(qBConv3rd_t), cudaHostAllocMapped);
}

void DCE::cleanMem_ZeroCopy() {
    if (INDATA != NULL)
        cudaFree(INDATA);
    if (OUTDATA != NULL)
        cudaFree(OUTDATA);
    if (NETIO != NULL)
        cudaFree(NETIO);
    if (UPSBUFFER != NULL)
        cudaFree(UPSBUFFER);
    if (PARAM != NULL)
        cudaFree(PARAM);
    
    if (CONVW01 != NULL)
        cudaFree(CONVW01);
    if (CONVB01 != NULL)
        cudaFree(CONVB01);
    if (CONVW02 != NULL)
        cudaFree(CONVW02);
    if (CONVB02 != NULL)
        cudaFree(CONVB02);
    if (CONVW03 != NULL)
        cudaFree(CONVW03);
    if (CONVB03 != NULL)
        cudaFree(CONVB03);
}

void DCE::cleanMem() {
    // Free CPU Memory
    if (INDATA != NULL) {
        free(INDATA);
    }
    if (OUTDATA != NULL) {
        free(OUTDATA);
    }
}

void DCE::loadWeight() {
    memcpy((void*)CONVW01, conv1_w, 2 * 864);
    memcpy((void*)CONVB01, conv1_b, 4 * 32);
    memcpy((void*)CONVW02, conv2_w, 2 * 9216);
    memcpy((void*)CONVB02, conv2_b, 4 * 32);
    memcpy((void*)CONVW03, conv3_w, 2 * 864);
    memcpy((void*)CONVB03, conv3_b, 4 * 3);
}

__global__ void dNorm(RGBIOData_t* dINDATA, qNormImg_t* dNORM) {
    unsigned int globalIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int globalIdx_y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int globalIdx_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    dNORM->data[globalIdx_x][globalIdx_y][globalIdx_z] = (short)dINDATA->data[globalIdx_x][globalIdx_y][globalIdx_z] << 6;
}

__global__ void dDownSample(qNormImg_t* dNORM, qNetIO_t* dNETIO)
{
    unsigned int globalIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int globalIdx_y = blockIdx.y * blockDim.y + threadIdx.y;

    dNETIO->data[globalIdx_x][globalIdx_y][0] = dNORM->data[globalIdx_x * DSRATE][globalIdx_y * DSRATE][0];
    dNETIO->data[globalIdx_x][globalIdx_y][1] = dNORM->data[globalIdx_x * DSRATE][globalIdx_y * DSRATE][1];
    dNETIO->data[globalIdx_x][globalIdx_y][2] = dNORM->data[globalIdx_x * DSRATE][globalIdx_y * DSRATE][2];
}

void DCE::qNormNDownSample_ZeroCopy() {
    cudaError_t error;

    error = cudaHostGetDevicePointer((void**)&dINDATA,  (void*)INDATA , 0);
    if (error != cudaSuccess) {
        printf("Error dINDATA cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&dNORM, sizeof(qNormImg_t));
    if (error != cudaSuccess) {
        printf("Error dNORM cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    // Run dNorm kernel
    //constexpr unsigned int jobSize = IMG_HIGHT * IMG_WIDTH * IMG_CHANNEL;

    dim3 dimBlock {24, 40, 1};
    dim3 dimGrid;

    dimGrid.x = (IMG_HIGHT + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (IMG_WIDTH + dimBlock.y - 1) / dimBlock.y;
    dimGrid.z = (IMG_CHANNEL + dimBlock.z - 1) / dimBlock.z;

    //printf("dimBlock {%d, %d, %d} dimGrid {%d, %d, %d}\n\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);

    dNorm<<<dimGrid, dimBlock>>>(dINDATA, dNORM);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dNorm %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer((void**)&dNETIO,  (void*)NETIO , 0);
    if (error != cudaSuccess) {
        printf("Error dNETIO cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    // Run dDownSample kernel
    dimBlock = {30, 32, 1};
    dimGrid = {DCE_HEIGHT / dimBlock.x, DCE_WIDTH / dimBlock.y, 1};

    dDownSample<<<dimGrid, dimBlock>>>(dNORM, dNETIO);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dDownSample %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void DCE::qNormNDownSample() {
    cudaError_t error;

    error = cudaMalloc(&dINDATA, sizeof(RGBIOData_t));
    if (error != cudaSuccess) {
        printf("Error dINDATA cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(dINDATA, INDATA, sizeof(RGBIOData_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dINDATA cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    error = cudaMalloc(&dNORM, sizeof(qNormImg_t));
    if (error != cudaSuccess) {
        printf("Error dNORM cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    // Run dNorm kernel
    //constexpr unsigned int jobSize = IMG_HIGHT * IMG_WIDTH * IMG_CHANNEL;

    dim3 dimBlock {24, 40, 1};
    dim3 dimGrid;

    dimGrid.x = (IMG_HIGHT + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (IMG_WIDTH + dimBlock.y - 1) / dimBlock.y;
    dimGrid.z = (IMG_CHANNEL + dimBlock.z - 1) / dimBlock.z;

    //printf("dimBlock {%d, %d, %d} dimGrid {%d, %d, %d}\n\n", dimBlock.x, dimBlock.y, dimBlock.z, dimGrid.x, dimGrid.y, dimGrid.z);

    dNorm<<<dimGrid, dimBlock>>>(dINDATA, dNORM);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dNorm %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

    if (dINDATA != nullptr) {
        error = cudaFree(dINDATA);
        if (error != cudaSuccess) {
            printf("Error dINDATA cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }

    error = cudaMalloc(&dNETIO, sizeof(qNetIO_t));
    if (error != cudaSuccess) {
        printf("Error dNETIO cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    // Run dDownSample kernel
    dimBlock = {30, 32, 1};
    dimGrid = {DCE_HEIGHT / dimBlock.x, DCE_WIDTH / dimBlock.y, 1};

    dDownSample<<<dimGrid, dimBlock>>>(dNORM, dNETIO);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dDownSample %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

__global__ void dConv1st(qNetIO_t* dNETIO, qWConv1st_t* dCONVW01, qBConv1st_t* dCONVB01, qNetFeature_t* dFEATURE1) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    int sum;
    for (int cout = 0; cout < DCE_CHANNEL; ++cout) {
        sum = 0;
        for (int cin = 0; cin < IMG_CHANNEL; ++cin) {
            for (int kh = -1; kh <= 1; ++kh) {
                for (int kw = -1; kw <= 1; ++kw) {
                    if(((h + kh) >= 0) && ((w + kw) >= 0) && ((h + kh) < DCE_HEIGHT) && ((w + kw) < DCE_WIDTH))
                        sum += dNETIO->data[h + kh][w + kw][cin] * dCONVW01->data[cout][cin][kh + 1][kw + 1];
                }
            }
        }
        sum += dCONVB01->data[cout];
        sum = max(0, sum);
        dFEATURE1->data[h][w][cout] = sum >> 14;
    }
}

void DCE::qConv1st_ZeroCopy() {
    cudaError_t error;

    error = cudaHostGetDevicePointer((void**)&dCONVW01,  (void*)CONVW01 , 0);
    if (error != cudaSuccess) {
        printf("Error dCONVW01 cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer((void**)&dCONVB01,  (void*)CONVB01 , 0);
    if (error != cudaSuccess) {
        printf("Error dCONVB01 cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&dFEATURE1, sizeof(qNetFeature_t));
    if (error != cudaSuccess) {
        printf("Error dFEATURE1 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Run dConv1st kernel
    dim3 dimBlock = {18, 40, 1};
    dim3 dimGrid = {DCE_HEIGHT / dimBlock.x, DCE_WIDTH / dimBlock.y, 1};

    dConv1st<<<dimGrid, dimBlock>>>(dNETIO, dCONVW01, dCONVB01, dFEATURE1);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dConv1st %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void DCE::qConv1st() {
    cudaError_t error;

    error = cudaMalloc(&dCONVW01, sizeof(qWConv1st_t));
    if (error != cudaSuccess) {
        printf("Error dCONVW01 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(dCONVW01, CONVW01, sizeof(qWConv1st_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dCONVW01 cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    error = cudaMalloc(&dCONVB01, sizeof(qBConv1st_t));
    if (error != cudaSuccess) {
        printf("Error dCONVB01 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(dCONVB01, CONVB01, sizeof(qBConv1st_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dCONVB01 cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    error = cudaMalloc(&dFEATURE1, sizeof(qNetFeature_t));
    if (error != cudaSuccess) {
        printf("Error dFEATURE1 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Run dConv1st kernel
    dim3 dimBlock = {18, 40, 1};
    dim3 dimGrid = {DCE_HEIGHT / dimBlock.x, DCE_WIDTH / dimBlock.y, 1};

    dConv1st<<<dimGrid, dimBlock>>>(dNETIO, dCONVW01, dCONVB01, dFEATURE1);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dConv1st %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

    // Free CONVW01, CONVB01
    if (CONVW01 != nullptr)
        free(CONVW01);
    if (CONVB01 != nullptr)
        free(CONVB01);

    // Free dCONVW01, dCONVB01
    if (dCONVW01 != nullptr) {
        error = cudaFree(dCONVW01);
        if (error != cudaSuccess) {
            printf("Error dCONVW01 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
    if (dCONVB01 != nullptr) {
        error = cudaFree(dCONVB01);
        if (error != cudaSuccess) {
            printf("Error dCONVB01 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
}

__global__ void dConv2nd(qNetFeature_t* dFEATURE1, qWConv2nd_t* dCONVW02, qBConv2nd_t* dCONVB02, qNetFeature_t* dFEATURE2) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    int sum;
    for (int cout = 0; cout < DCE_CHANNEL; ++cout) {
        sum = 0;
        for (int cin = 0; cin < DCE_CHANNEL; ++cin) {
            for (int kh = -1; kh <= 1; ++kh) {
                for (int kw = -1; kw <= 1; ++kw) {
                    if(((h + kh) >= 0) && ((w + kw) >= 0) && ((h + kh) < DCE_HEIGHT) && ((w + kw) < DCE_WIDTH))
                        sum += dFEATURE1->data[h + kh][w + kw][cin] * dCONVW02->data[cout][cin][kh + 1][kw + 1];
                }
            }
        }
        sum += dCONVB02->data[cout];
        sum = max(0, sum);
        dFEATURE2->data[h][w][cout] = sum >> 14;
    }
}

void DCE::qConv2nd_ZeroCopy() {
    cudaError_t error;

    error = cudaHostGetDevicePointer((void**)&dCONVW02,  (void*)CONVW02 , 0);
    if (error != cudaSuccess) {
        printf("Error dCONVW02 cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer((void**)&dCONVB02,  (void*)CONVB02 , 0);
    if (error != cudaSuccess) {
        printf("Error dCONVB02 cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&dFEATURE2, sizeof(qNetFeature_t));
    if (error != cudaSuccess) {
        printf("Error dFEATURE2 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Run dConv1st kernel
    dim3 dimBlock = {18, 40, 1};
    dim3 dimGrid = {DCE_HEIGHT / dimBlock.x, DCE_WIDTH / dimBlock.y, 1};

    dConv2nd<<<dimGrid, dimBlock>>>(dFEATURE1, dCONVW02, dCONVB02, dFEATURE2);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dConv2nd %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void DCE::qConv2nd() {
    cudaError_t error;

    error = cudaMalloc(&dCONVW02, sizeof(qWConv2nd_t));
    if (error != cudaSuccess) {
        printf("Error dCONVW02 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(dCONVW02, CONVW02, sizeof(qWConv2nd_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dCONVW02 cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    error = cudaMalloc(&dCONVB02, sizeof(qBConv2nd_t));
    if (error != cudaSuccess) {
        printf("Error dCONVB02 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(dCONVB02, CONVB02, sizeof(qBConv2nd_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dCONVB02 cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    error = cudaMalloc(&dFEATURE2, sizeof(qNetFeature_t));
    if (error != cudaSuccess) {
        printf("Error dFEATURE2 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Run dConv1st kernel
    dim3 dimBlock = {18, 40, 1};
    dim3 dimGrid = {DCE_HEIGHT / dimBlock.x, DCE_WIDTH / dimBlock.y, 1};

    dConv2nd<<<dimGrid, dimBlock>>>(dFEATURE1, dCONVW02, dCONVB02, dFEATURE2);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dConv2nd %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

    // Free CONVW01, CONVB01
    if (CONVW02 != nullptr)
        free(CONVW02);
    if (CONVB02 != nullptr)
        free(CONVB02);

    // Free dCONVW01, dCONVB01
    if (dCONVW02 != nullptr) {
        error = cudaFree(dCONVW02);
        if (error != cudaSuccess) {
            printf("Error dCONVW02 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
    if (dCONVB02 != nullptr) {
        error = cudaFree(dCONVB02);
        if (error != cudaSuccess) {
            printf("Error dCONVB02 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
}

__device__ int sigmoidMapping(int x)
{
    if (x >= -1 * QX && x <= QX)
        return (x * 7810 + 195) >> 18;
    else if ((x >= -2 * QX && x < -1 * QX) || (x <= 2 * QX && x > QX))
        return x > 0 ? ((x * 4899 + 47996260) >> 18) : ((x * 4899 - 47996260) >> 18);
    else if ((x >= -3 * QX && x < -2 * QX) || (x <= 3 * QX && x > 2 * QX))
        return x > 0 ? ((x * 2330 + 130915972) >> 18) : ((x * 2330 - 130915972) >> 18);
    else if ((x >= -4 * QX && x < -3 * QX) || (x <= 4 * QX && x > 3 * QX))
        return x > 0 ? ((x * 952 + 197514809) >> 18) : ((x * 952 - 197514809) >> 18);
    else if ((x >= -5 * QX && x < -4 * QX) || (x <= 5 * QX && x > 4 * QX))
        return x > 0 ? ((x * 364 + 235417895) >> 18) : ((x * 364 - 235417895) >> 18);
    else
        return (x > 0 ? QA : -QA);
}

__global__ void dConv3rd(qNetFeature_t* dFEATURE1, qNetFeature_t* dFEATURE2, qWConv3rd_t* dCONVW03, qBConv3rd_t* dCONVB03, qNetIO_t* dNETIO) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    int sum;
    for (int cout = 0; cout < IMG_CHANNEL; ++cout) {
        sum = 0;
        for (int cin = 0; cin < DCE_CHANNEL; ++cin) {
            for (int kh = -1; kh <= 1; ++kh) {
                for (int kw = -1; kw <= 1; ++kw) {
                    if(((h + kh) >= 0) && ((w + kw) >= 0) && ((h + kh) < DCE_HEIGHT) && ((w + kw) < DCE_WIDTH))
                        sum += (dFEATURE1->data[h + kh][w + kw][cin] + dFEATURE2->data[h + kh][w + kw][cin]) * dCONVW03->data[cout][cin][kh + 1][kw + 1];
                }
            }
        }
        sum += dCONVB03->data[cout];
        dNETIO->data[h][w][cout] = (short)sigmoidMapping(sum >> 14);
    }
}

void DCE::qConv3rd_ZeroCopy() {
    cudaError_t error;

    error = cudaHostGetDevicePointer((void**)&dCONVW03,  (void*)CONVW03 , 0);
    if (error != cudaSuccess) {
        printf("Error dCONVW03 cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer((void**)&dCONVB03,  (void*)CONVB03 , 0);
    if (error != cudaSuccess) {
        printf("Error dCONVB03 cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Run dConv3rd
    dim3 dimBlock = {18, 40, 1};
    dim3 dimGrid = {DCE_HEIGHT / dimBlock.x, DCE_WIDTH / dimBlock.y, 1};
    dConv3rd<<<dimGrid, dimBlock>>>(dFEATURE1, dFEATURE2, dCONVW03, dCONVB03, dNETIO);

    if (dFEATURE1 != nullptr) {
        error = cudaFree(dFEATURE1);
        if (error != cudaSuccess) {
            printf("Error dFEATURE1 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
    if (dFEATURE2 != nullptr) {
        error = cudaFree(dFEATURE2);
        if (error != cudaSuccess) {
            printf("Error dFEATURE2 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
}

void DCE::qConv3rd() {
    cudaError_t error;

    error = cudaMalloc(&dCONVW03, sizeof(qWConv3rd_t));
    if (error != cudaSuccess) {
        printf("Error dCONVW03 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(dCONVW03, CONVW03, sizeof(qWConv3rd_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dCONVW03 cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    error = cudaMalloc(&dCONVB03, sizeof(qBConv3rd_t));
    if (error != cudaSuccess) {
        printf("Error dCONVB03 cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(dCONVB03, CONVB03, sizeof(qBConv3rd_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dCONVB03 cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    // Run dConv3rd
    dim3 dimBlock = {18, 40, 1};
    dim3 dimGrid = {DCE_HEIGHT / dimBlock.x, DCE_WIDTH / dimBlock.y, 1};
    dConv3rd<<<dimGrid, dimBlock>>>(dFEATURE1, dFEATURE2, dCONVW03, dCONVB03, dNETIO);

    // Release dFEATURE1 dFEATURE2 dCONVW3 dCONVB3
    if (CONVW03 != nullptr)
        free(CONVW03);
    if (CONVB03 != nullptr)
        free(CONVB03);

    if (dCONVW03 != nullptr) {
        error = cudaFree(dCONVW03);
        if (error != cudaSuccess) {
            printf("Error dCONVW03 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
    if (dCONVB03 != nullptr) {
        error = cudaFree(dCONVB03);
        if (error != cudaSuccess) {
            printf("Error dCONVB03 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
    if (dFEATURE1 != nullptr) {
        error = cudaFree(dFEATURE1);
        if (error != cudaSuccess) {
            printf("Error dFEATURE1 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
    if (dFEATURE2 != nullptr) {
        error = cudaFree(dFEATURE2);
        if (error != cudaSuccess) {
            printf("Error dFEATURE2 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }

    // Copy NETIO to CPU
    error = cudaMemcpy(NETIO, dNETIO, sizeof(qNetIO_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error dNETIO cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    // Release NETIO
    if (dNETIO != nullptr) {
        error = cudaFree(dNETIO);
        if (error != cudaSuccess) {
            printf("Error dNETIO cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }
}

void DCE::qUpSample_ZeroCopy() {
    int coef[12] = {42, 128, 213, 298, 384, 469, 554, 640, 725, 810, 896, 981};
	for(int h = 0; h < DCE_HEIGHT; ++h)
	{
		int wi = 0;
		for(int d = 0; d < DSRATE / 2; ++d, ++wi)
		{
			UPSBUFFER->data[h][wi][0] = NETIO->data[h][0][0];
			UPSBUFFER->data[h][wi][1] = NETIO->data[h][0][1];
			UPSBUFFER->data[h][wi][2] = NETIO->data[h][0][2];
		}
		for(int w = 1; w < DCE_WIDTH; ++w)
		{
			for(int d = 0; d < DSRATE; ++d, ++wi)
			{
				UPSBUFFER->data[h][wi][0] = (coef[d] * (NETIO->data[h][w][0] - NETIO->data[h][w - 1][0]) >> 10) + NETIO->data[h][w - 1][0];
				UPSBUFFER->data[h][wi][1] = (coef[d] * (NETIO->data[h][w][1] - NETIO->data[h][w - 1][1]) >> 10) + NETIO->data[h][w - 1][1];
				UPSBUFFER->data[h][wi][2] = (coef[d] * (NETIO->data[h][w][2] - NETIO->data[h][w - 1][2]) >> 10) + NETIO->data[h][w - 1][2];
			}
		}
		for(int d = 0; d < DSRATE / 2; ++d, ++wi)
		{
			UPSBUFFER->data[h][wi][0] = NETIO->data[h][DCE_WIDTH - 1][0];
			UPSBUFFER->data[h][wi][1] = NETIO->data[h][DCE_WIDTH - 1][1];
			UPSBUFFER->data[h][wi][2] = NETIO->data[h][DCE_WIDTH - 1][2];
		}
	}

	int hi = 0;
	for(int d = 0; d < DSRATE / 2; ++d, ++hi)
	{
		for(int w = 0; w < IMG_WIDTH; ++w)
		{
			PARAM->data[hi][w][0] = UPSBUFFER->data[0][w][0];
			PARAM->data[hi][w][1] = UPSBUFFER->data[0][w][1];
			PARAM->data[hi][w][2] = UPSBUFFER->data[0][w][2];
		}
	}
	for(int h = 1; h < DCE_HEIGHT; ++h)
	{
		for(int d = 0; d < DSRATE; ++d, ++hi)
		{
			for(int w = 0; w < IMG_WIDTH; ++w)
			{
				PARAM->data[hi][w][0] = (coef[d] * (UPSBUFFER->data[h][w][0] - UPSBUFFER->data[h - 1][w][0]) >> 10) + UPSBUFFER->data[h - 1][w][0];
				PARAM->data[hi][w][1] = (coef[d] * (UPSBUFFER->data[h][w][1] - UPSBUFFER->data[h - 1][w][1]) >> 10) + UPSBUFFER->data[h - 1][w][1];
				PARAM->data[hi][w][2] = (coef[d] * (UPSBUFFER->data[h][w][2] - UPSBUFFER->data[h - 1][w][2]) >> 10) + UPSBUFFER->data[h - 1][w][2];
			}

		}
	}
	for(int d = 0; d < DSRATE / 2; ++d, ++hi)
	{
		for(int w = 0; w < IMG_WIDTH; ++w)
		{
			PARAM->data[hi][w][0] = UPSBUFFER->data[DCE_HEIGHT - 1][w][0];
			PARAM->data[hi][w][1] = UPSBUFFER->data[DCE_HEIGHT - 1][w][1];
			PARAM->data[hi][w][2] = UPSBUFFER->data[DCE_HEIGHT - 1][w][2];
		}
	}
}

void DCE::qUpSample()
{
    int coef[12] = {42, 128, 213, 298, 384, 469, 554, 640, 725, 810, 896, 981};
	for(int h = 0; h < DCE_HEIGHT; ++h)
	{
		int wi = 0;
		for(int d = 0; d < DSRATE / 2; ++d, ++wi)
		{
			UPSBUFFER->data[h][wi][0] = NETIO->data[h][0][0];
			UPSBUFFER->data[h][wi][1] = NETIO->data[h][0][1];
			UPSBUFFER->data[h][wi][2] = NETIO->data[h][0][2];
		}
		for(int w = 1; w < DCE_WIDTH; ++w)
		{
			for(int d = 0; d < DSRATE; ++d, ++wi)
			{
				UPSBUFFER->data[h][wi][0] = (coef[d] * (NETIO->data[h][w][0] - NETIO->data[h][w - 1][0]) >> 10) + NETIO->data[h][w - 1][0];
				UPSBUFFER->data[h][wi][1] = (coef[d] * (NETIO->data[h][w][1] - NETIO->data[h][w - 1][1]) >> 10) + NETIO->data[h][w - 1][1];
				UPSBUFFER->data[h][wi][2] = (coef[d] * (NETIO->data[h][w][2] - NETIO->data[h][w - 1][2]) >> 10) + NETIO->data[h][w - 1][2];
			}
		}
		for(int d = 0; d < DSRATE / 2; ++d, ++wi)
		{
			UPSBUFFER->data[h][wi][0] = NETIO->data[h][DCE_WIDTH - 1][0];
			UPSBUFFER->data[h][wi][1] = NETIO->data[h][DCE_WIDTH - 1][1];
			UPSBUFFER->data[h][wi][2] = NETIO->data[h][DCE_WIDTH - 1][2];
		}
	}

	int hi = 0;
	for(int d = 0; d < DSRATE / 2; ++d, ++hi)
	{
		for(int w = 0; w < IMG_WIDTH; ++w)
		{
			PARAM->data[hi][w][0] = UPSBUFFER->data[0][w][0];
			PARAM->data[hi][w][1] = UPSBUFFER->data[0][w][1];
			PARAM->data[hi][w][2] = UPSBUFFER->data[0][w][2];
		}
	}
	for(int h = 1; h < DCE_HEIGHT; ++h)
	{
		for(int d = 0; d < DSRATE; ++d, ++hi)
		{
			for(int w = 0; w < IMG_WIDTH; ++w)
			{
				PARAM->data[hi][w][0] = (coef[d] * (UPSBUFFER->data[h][w][0] - UPSBUFFER->data[h - 1][w][0]) >> 10) + UPSBUFFER->data[h - 1][w][0];
				PARAM->data[hi][w][1] = (coef[d] * (UPSBUFFER->data[h][w][1] - UPSBUFFER->data[h - 1][w][1]) >> 10) + UPSBUFFER->data[h - 1][w][1];
				PARAM->data[hi][w][2] = (coef[d] * (UPSBUFFER->data[h][w][2] - UPSBUFFER->data[h - 1][w][2]) >> 10) + UPSBUFFER->data[h - 1][w][2];
			}

		}
	}
	for(int d = 0; d < DSRATE / 2; ++d, ++hi)
	{
		for(int w = 0; w < IMG_WIDTH; ++w)
		{
			PARAM->data[hi][w][0] = UPSBUFFER->data[DCE_HEIGHT - 1][w][0];
			PARAM->data[hi][w][1] = UPSBUFFER->data[DCE_HEIGHT - 1][w][1];
			PARAM->data[hi][w][2] = UPSBUFFER->data[DCE_HEIGHT - 1][w][2];
		}
	}

    if (UPSBUFFER != nullptr)
        free(UPSBUFFER);
    if (NETIO != nullptr)
        free(NETIO);
}

__global__ void dEnhance(qNormImg_t* dNORM, qEnhancedParam_t* dPARAM, RGBIOData_t* dOUTDATA)
{
    int globalIdx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int globalIdx_z = blockIdx.z * blockDim.z + threadIdx.z;

    int qX = dNORM->data[globalIdx_x][globalIdx_y][globalIdx_z] >> 4;
    int qP = dPARAM->data[globalIdx_x][globalIdx_y][globalIdx_z];

    int output;
    for (int i =0; i < 8; ++i) {
        int qX2 = qX << 10;
        int qX3 = qX2 << 10;
        qX3 = qX3 + qP * (qX * qX - qX2);
        qX = (int)(qX3 >> 20);
    }
    output = qX >> 2;
    output = output > 255 ? 255 : output;

    dOUTDATA->data[globalIdx_x][globalIdx_y][globalIdx_z] = (uint8_t)output;
}

void DCE::qEnhance_ZeroCopy()
{
    cudaError_t error;

    error = cudaHostGetDevicePointer((void**)&dPARAM,  (void*)PARAM , 0);
    if (error != cudaSuccess) {
        printf("Error dPARAM cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaHostGetDevicePointer((void**)&dOUTDATA,  (void*)OUTDATA , 0);
    if (error != cudaSuccess) {
        printf("Error dOUTDATA cudaHostGetDevicePointer() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    dim3 dimBlock {24, 30, 1};
    dim3 dimGrid;

    dimGrid.x = (IMG_HIGHT + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (IMG_WIDTH + dimBlock.y - 1) / dimBlock.y;
    dimGrid.z = (IMG_CHANNEL + dimBlock.z - 1) / dimBlock.z;

    dEnhance<<<dimGrid, dimBlock>>>(dNORM, dPARAM, dOUTDATA);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dEnhance %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void DCE::qEnhance()
{
    cudaError_t error;

    // Copy PARAM to GPU
    error = cudaMalloc(&dPARAM, sizeof(qEnhancedParam_t));
    if (error != cudaSuccess) {
        printf("Error dPARAM cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(dPARAM, PARAM, sizeof(qEnhancedParam_t), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Error dPARAM cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }  

    // Malloc dOUTDATA
    error = cudaMalloc(&dOUTDATA, sizeof(RGBIOData_t));
    if (error != cudaSuccess) {
        printf("Error dOUTDATA cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    dim3 dimBlock {24, 30, 1};
    dim3 dimGrid;

    dimGrid.x = (IMG_HIGHT + dimBlock.x - 1) / dimBlock.x;
    dimGrid.y = (IMG_WIDTH + dimBlock.y - 1) / dimBlock.y;
    dimGrid.z = (IMG_CHANNEL + dimBlock.z - 1) / dimBlock.z;

    dEnhance<<<dimGrid, dimBlock>>>(dNORM, dPARAM, dOUTDATA);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dEnhance %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

    error = cudaMemcpy(OUTDATA, dOUTDATA, sizeof(RGBIOData_t), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        printf("Error dOUTDATA cudaMemcpy() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);  
    }

    // Free dOUTDATA, dPARAM, dNORM
    if (dOUTDATA != nullptr) {
        error = cudaFree(dOUTDATA);
        if (error != cudaSuccess) {
            printf("Error dOUTDATA cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }

    if (dPARAM != nullptr) {
        error = cudaFree(dPARAM);
        if (error != cudaSuccess) {
            printf("Error dPARAM cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }

    if (dNORM != nullptr) {
        error = cudaFree(dNORM);
        if (error != cudaSuccess) {
            printf("Error dNORM cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
    }

    // Free PARAM
    if (PARAM != nullptr)
        free(PARAM);
}

void cvf::cvReadImg(char* filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_COLOR);
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    //cv::resize(image, image, cv::Size(1920, 1200), cv::INTER_AREA);
    #pragma omp parallel for schedule(guided) num_threads(NTHREAD)
    for (int c = 0; c < IMG_CHANNEL; c++) {
        for (int y = 0; y < IMG_HIGHT; ++y) {
            for(int x = 0; x < IMG_WIDTH; ++x) {
                INDATA->data[y][x][c] = image.at<cv::Vec3b>(y, x)[c];
            }
        }
    }
    //cv::waitKey(0);
}

void cvf::cvOutputImg(char* filename) {
    cv::Mat image(IMG_HIGHT, IMG_WIDTH, CV_8UC3);

    #pragma omp parallel for schedule(guided) num_threads(NTHREAD)
    for (int c = 0; c < IMG_CHANNEL; c++)
        for (int y = 0; y < IMG_HIGHT; ++y)
            for (int x = 0; x < IMG_WIDTH; ++x) {
                image.at<cv::Vec3b>(y, x)[c] = OUTDATA->data[y][x][c];
            }
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    cv::imwrite(filename, image);

    //cv::waitKey(0);
}

std::string cvf::gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method) {
    return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(capture_width) + ", height=(int)" +
           std::to_string(capture_height) + ", framerate=(fraction)" + std::to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + std::to_string(flip_method) + " ! video/x-raw, width=(int)" + std::to_string(display_width) + ", height=(int)" +
           std::to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}
