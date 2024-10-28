#include "main.cuh"

void DCE::initMem() {
    INDATA = (RGBIOData*)malloc(sizeof(RGBIOData_t));
    
    OUTDATA = (RGBIOData*)malloc(sizeof(RGBIOData_t));

    NETIO = (qNetIO_t*)malloc(sizeof(qNetIO_t));
#ifdef CPU_UPSAMPLE
    UPSBUFFER = (qEnhancedParam_t*)malloc(sizeof(qEnhancedParam_t));
    PARAM = (qEnhancedParam_t*)malloc(sizeof(qEnhancedParam_t));
#endif
    CONVW01 = (qWConv1st_t*)malloc(sizeof(qWConv1st_t));
    CONVB01 = (qBConv1st_t*)malloc(sizeof(qBConv1st_t));
    CONVW02 = (qWConv2nd_t*)malloc(sizeof(qWConv2nd_t));
    CONVB02 = (qBConv2nd_t*)malloc(sizeof(qBConv2nd_t));
    CONVW03 = (qWConv3rd_t*)malloc(sizeof(qWConv3rd_t));
    CONVB03 = (qBConv3rd_t*)malloc(sizeof(qBConv3rd_t));
}

void DCE::cleanMem() {
    // Free CPU Memory
    if (INDATA != nullptr) {
        free(INDATA);
        INDATA = nullptr;
    }
    if (OUTDATA != nullptr) {
        free(OUTDATA);
        OUTDATA = nullptr;
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
        dINDATA = nullptr;
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
    if (CONVW01 != nullptr) {
        free(CONVW01);
        CONVW01 = nullptr;
    }
    if (CONVB01 != nullptr) {
        free(CONVB01);
        CONVB01 = nullptr;
    }

    // Free dCONVW01, dCONVB01
    if (dCONVW01 != nullptr) {
        error = cudaFree(dCONVW01);
        if (error != cudaSuccess) {
            printf("Error dCONVW01 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dCONVW01 = nullptr;
    }
    if (dCONVB01 != nullptr) {
        error = cudaFree(dCONVB01);
        if (error != cudaSuccess) {
            printf("Error dCONVB01 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dCONVB01 = nullptr;
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
    if (CONVW02 != nullptr) {
        free(CONVW02);
        CONVW02 = nullptr;
    }
    if (CONVB02 != nullptr) {
        free(CONVB02);
        CONVB02 = nullptr;
    }

    // Free dCONVW01, dCONVB01
    if (dCONVW02 != nullptr) {
        error = cudaFree(dCONVW02);
        if (error != cudaSuccess) {
            printf("Error dCONVW02 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dCONVW02 = nullptr;
    }
    if (dCONVB02 != nullptr) {
        error = cudaFree(dCONVB02);
        if (error != cudaSuccess) {
            printf("Error dCONVB02 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dCONVB02 = nullptr;
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
    if (CONVW03 != nullptr) {
        free(CONVW03);
        CONVW03 = nullptr;
    }
    if (CONVB03 != nullptr) {
        free(CONVB03);
        CONVB03 = nullptr;
    }

    if (dCONVW03 != nullptr) {
        error = cudaFree(dCONVW03);
        if (error != cudaSuccess) {
            printf("Error dCONVW03 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dCONVW03 = nullptr;
    }
    if (dCONVB03 != nullptr) {
        error = cudaFree(dCONVB03);
        if (error != cudaSuccess) {
            printf("Error dCONVB03 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dCONVB03 = nullptr;
    }
    if (dFEATURE1 != nullptr) {
        error = cudaFree(dFEATURE1);
        if (error != cudaSuccess) {
            printf("Error dFEATURE1 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dFEATURE1 = nullptr;
    }
    if (dFEATURE2 != nullptr) {
        error = cudaFree(dFEATURE2);
        if (error != cudaSuccess) {
            printf("Error dFEATURE2 cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dFEATURE2 = nullptr;
    }

    // Copy NETIO to CPU
#ifdef CPU_UPSAMPLE
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
        dNETIO = nullptr;
    }
#endif
}

#ifdef CPU_UPSAMPLE
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

    if (UPSBUFFER != nullptr) {
        free(UPSBUFFER);
        UPSBUFFER = nullptr;
    }
    if (NETIO != nullptr) {
        free(NETIO);
        NETIO = nullptr;
    }
}
#else
void DCE::qUpSample() {
    cudaError_t error;

    error = cudaMalloc(&dUPSBUFFER, sizeof(qEnhancedParam_t));
    if (error != cudaSuccess) {
        printf("Error dUPSBUFFER cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&dPARAM, sizeof(qEnhancedParam_t));
    if (error != cudaSuccess) {
        printf("Error dPARAM cudaMalloc() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Run dUpSample_x
    dim3 dimBlock = {30, 30, 1};
    dim3 dimGrid = {DCE_HEIGHT / dimBlock.x, IMG_WIDTH / dimBlock.y, 1};    
    dUpSample_x<<<dimGrid, dimBlock>>>(dNETIO, dUPSBUFFER);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dUpSample_x %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

    // Run dUpSample_y
    dimBlock = {24, 30, 1};
    dimGrid = {IMG_HIGHT / dimBlock.x, IMG_WIDTH / dimBlock.y, 1};
    dUpSample_y<<<dimGrid, dimBlock>>>(dUPSBUFFER, dPARAM);

    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
		printf("Error: dUpSample_y %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

    if (dUPSBUFFER != nullptr) {
        error = cudaFree(dUPSBUFFER);
        if (error != cudaSuccess) {
            printf("Error dUPSBUFFER cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dUPSBUFFER = nullptr;
    }
}
#endif

void DCE::qEnhance()
{
    cudaError_t error;

#ifdef CPU_UPSAMPLE
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
#endif

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
        dOUTDATA = nullptr;
    }

    if (dPARAM != nullptr) {
        error = cudaFree(dPARAM);
        if (error != cudaSuccess) {
            printf("Error dPARAM cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dPARAM = nullptr;
    }

    if (dNORM != nullptr) {
        error = cudaFree(dNORM);
        if (error != cudaSuccess) {
            printf("Error dNORM cudaFree() : %d\n%s\n\n", static_cast<int>(error), cudaGetErrorString(error));
            exit(EXIT_FAILURE);  
        }
        dNORM = nullptr;
    }

    // Free PARAM
    if (PARAM != nullptr) {
        free(PARAM);
        PARAM = nullptr;
    }
}
