#include "main.cuh"

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

// Notice that h -> DCE_HEIGHT w -> IMG_WIDTH
// coef[12] {42, 128, 213, 298, 384, 469, 554, 640, 725, 810, 896, 981}
__global__ void dUpSample_x(qNetIO_t* dNETIO, qEnhancedParam_t* dUPSBUFFER) // TODO modified entire program to fit into this modification
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr int pad = DSRATE / 2;

    __shared__ int coef[12];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        coef[0] = 42;
        coef[1] = 128;
        coef[2] = 213;
        coef[3] = 298;
        coef[4] = 384;
        coef[5] = 469;
        coef[6] = 554;
        coef[7] = 640;
        coef[8] = 725;
        coef[9] = 810;
        coef[10] = 896;
        coef[11] = 981;
    }

    __syncthreads();

    int hi = h / DSRATE;
    if (w < pad) {
        dUPSBUFFER->data[h][w][0] = dNETIO->data[hi][0][0];
        dUPSBUFFER->data[h][w][1] = dNETIO->data[hi][0][1];
        dUPSBUFFER->data[h][w][2] = dNETIO->data[hi][0][2];
    }
    else if (w >= (IMG_WIDTH - pad)) {
		dUPSBUFFER->data[h][w][0] = dNETIO->data[hi][DCE_WIDTH - 1][0];
		dUPSBUFFER->data[h][w][1] = dNETIO->data[hi][DCE_WIDTH - 1][1];
		dUPSBUFFER->data[h][w][2] = dNETIO->data[hi][DCE_WIDTH - 1][2];
    }
    else {
        int d = (w - pad) % DSRATE;
        int wi = (w - pad) / DSRATE + 1;
        dUPSBUFFER->data[h][w][0] = (coef[d] * (dNETIO->data[h][wi][0] - dNETIO->data[h][wi - 1][0]) >> 10) + dNETIO->data[h][wi - 1][0];
		dUPSBUFFER->data[h][w][1] = (coef[d] * (dNETIO->data[h][wi][1] - dNETIO->data[h][wi - 1][1]) >> 10) + dNETIO->data[h][wi - 1][1];
		dUPSBUFFER->data[h][w][2] = (coef[d] * (dNETIO->data[h][wi][2] - dNETIO->data[h][wi - 1][2]) >> 10) + dNETIO->data[h][wi - 1][2];
    }
}

// Notice that h -> IMG_HEIGHT w -> IMG_WIDTH
__global__ void dUpSample_y(qEnhancedParam_t* dUPSBUFFER, qEnhancedParam_t* dPARAM) // TODO modified entire program to fit into this modification
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    int w = blockIdx.y * blockDim.y + threadIdx.y;

    constexpr int pad = DSRATE / 2;

    __shared__ int coef[12];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        coef[0] = 42;
        coef[1] = 128;
        coef[2] = 213;
        coef[3] = 298;
        coef[4] = 384;
        coef[5] = 469;
        coef[6] = 554;
        coef[7] = 640;
        coef[8] = 725;
        coef[9] = 810;
        coef[10] = 896;
        coef[11] = 981;
    }

    __syncthreads();

    if (h < pad) {
		dPARAM->data[h][w][0] = dUPSBUFFER->data[0][w][0];
		dPARAM->data[h][w][1] = dUPSBUFFER->data[0][w][1];
		dPARAM->data[h][w][2] = dUPSBUFFER->data[0][w][2];
    }
    if (h >= (IMG_HIGHT - pad)) {
		dPARAM->data[h][w][0] = dUPSBUFFER->data[DCE_HEIGHT - 1][w][0];
		dPARAM->data[h][w][1] = dUPSBUFFER->data[DCE_HEIGHT - 1][w][1];
		dPARAM->data[h][w][2] = dUPSBUFFER->data[DCE_HEIGHT - 1][w][2];
    }
    else {
        int d = (h - pad) % DSRATE;
        int hi = (h - pad) / DSRATE + 1;
		dPARAM->data[h][w][0] = (coef[d] * (dUPSBUFFER->data[hi][w][0] - dUPSBUFFER->data[hi - 1][w][0]) >> 10) + dUPSBUFFER->data[hi - 1][w][0];
		dPARAM->data[h][w][1] = (coef[d] * (dUPSBUFFER->data[hi][w][1] - dUPSBUFFER->data[hi - 1][w][1]) >> 10) + dUPSBUFFER->data[hi - 1][w][1];
		dPARAM->data[h][w][2] = (coef[d] * (dUPSBUFFER->data[hi][w][2] - dUPSBUFFER->data[hi - 1][w][2]) >> 10) + dUPSBUFFER->data[hi - 1][w][2];
    }
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
