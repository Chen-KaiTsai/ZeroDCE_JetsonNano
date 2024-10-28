#include "main.cuh"

void DCE_ZeroCopy::initMem_ZeroCopy() {
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

void DCE_ZeroCopy::cleanMem_ZeroCopy() {
    if (INDATA != nullptr)
        cudaFree(INDATA);
    if (OUTDATA != nullptr)
        cudaFree(OUTDATA);
    if (NETIO != nullptr)
        cudaFree(NETIO);
    if (UPSBUFFER != nullptr)
        cudaFree(UPSBUFFER);
    if (PARAM != nullptr)
        cudaFree(PARAM);
    
    if (CONVW01 != nullptr)
        cudaFree(CONVW01);
    if (CONVB01 != nullptr)
        cudaFree(CONVB01);
    if (CONVW02 != nullptr)
        cudaFree(CONVW02);
    if (CONVB02 != nullptr)
        cudaFree(CONVB02);
    if (CONVW03 != nullptr)
        cudaFree(CONVW03);
    if (CONVB03 != nullptr)
        cudaFree(CONVB03);
}

void DCE_ZeroCopy::qNormNDownSample_ZeroCopy() {
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

void DCE_ZeroCopy::qConv1st_ZeroCopy() {
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

void DCE_ZeroCopy::qConv2nd_ZeroCopy() {
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

void DCE_ZeroCopy::qConv3rd_ZeroCopy() {
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

void DCE_ZeroCopy::qUpSample_ZeroCopy() {
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

void DCE_ZeroCopy::qEnhance_ZeroCopy()
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
