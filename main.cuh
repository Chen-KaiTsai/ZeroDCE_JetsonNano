/**
 * 
 * @author Chen-Kai Tsai
 * @brief capture one image and enhance it
 * @version 0.5
 * @date 2024-4-26
 * 
 * @author Chen-Kai Tsai
 * @brief capture one image and enhance it
 * @version 0.6
 * @date 2024-4-28
 * @note Add not tested zero-copy version. Require further build check. Change thread block size for better performance
 * 
 * @ref https://github.com/JetsonHacksNano/CSI-Camera/blob/master/simple_camera.cpp
 * 
 * @author Chen-Kai Tsai
 * @brief capture one image from CSI camera and enhance its light condition
 * @version 0.7
 * @date 2024-5-18
 * @note Finish testing zero-copy version. Run slower but with less memory usage.
 * @note Finish kernel block size optimization
 * 
 * @author Chen-Kai Tsai
 * @brief Miner Fixs
 * @version 0.7.5
 * @date 2024-5-22
 * 
 * @author Chen-Kai Tsai
 * @brief Under construction of GPU version Up Sampling
 * @version 0.7.6
 * @note coef[12] = {42, 128, 213, 298, 384, 469, 554, 640, 725, 810, 896, 981};
 * @note This coef is centered and equal to {1/24 * 1024, 3/24 * 1024, 5/24 * 1024 ...}
 * @bug UPSBUFFER can be [DCE_HEIGHT][IMG_WIDTH][3]
 * @date 2024-10-28
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <string>

#include <cstdio>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cmath>

#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/logger.hpp>

// Configurations

constexpr unsigned int DSRATE = 12;
constexpr unsigned int IMG_HIGHT = 1080;
constexpr unsigned int IMG_WIDTH = 1920;
constexpr unsigned int IMG_CHANNEL = 3;
constexpr unsigned int DCE_HEIGHT = 1080 / DSRATE;
constexpr unsigned int DCE_WIDTH = 1920 / DSRATE;
constexpr unsigned int DCE_CHANNEL = 32;

#define QX 16384
#define QW 16384
#define QB (QX * QW)
#define QI 1024
#define QA 1024

#define NTHREAD 3

//#define ON_JETSON

void loadWeight();

namespace DCE
{
    void initMem();    
    void cleanMem();
    void qNormNDownSample();
    void qConv1st();
    void qConv2nd();
    void qConv3rd();
    void qUpSample();
    void qEnhance();
}

namespace DCE_ZeroCopy 
{
    void initMem_ZeroCopy();
    void cleanMem_ZeroCopy();
    void qNormNDownSample_ZeroCopy();
    void qConv1st_ZeroCopy();
    void qConv2nd_ZeroCopy();
    void qConv3rd_ZeroCopy();
    void qUpSample_ZeroCopy();
    void qEnhance_ZeroCopy();
}

namespace cvf
{
    void cvReadImg(char* filename);
    void cvOutputImg(char* filename);
    std::string gstreamer_pipeline (int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method);
}

/**
 * Grouping Data to Structure for Readability
 */

using RGBIOData_t = struct RGBIOData
{
    uint8_t data[IMG_HIGHT][IMG_WIDTH][IMG_CHANNEL];
};

using qNormImg_t = struct qNormImg
{
    short data[IMG_HIGHT][IMG_WIDTH][IMG_CHANNEL];
};

using qNetIO_t = struct qNetIO
{
    short data[DCE_HEIGHT][DCE_WIDTH][IMG_CHANNEL];
};

using qNetFeature_t = struct qNetFeature
{
    short data[DCE_HEIGHT][DCE_WIDTH][DCE_CHANNEL];
};

using qEnhancedParam_t = struct qEnhancedParam
{
    short data[IMG_HIGHT][IMG_WIDTH][IMG_CHANNEL];
};

using qWConv1st_t = struct qWConv1st
{
    short data[DCE_CHANNEL][IMG_CHANNEL][3][3];
};
using qBConv1st_t = struct qBConv1st
{
    int data[DCE_CHANNEL];
};

using qWConv2nd_t =  struct qWConv2nd
{
    short data[DCE_CHANNEL][DCE_CHANNEL][3][3];
};
using qBConv2nd_t = struct qBConv2nd
{
    int data[DCE_CHANNEL];
};

using qWConv3rd_t = struct qWConv3rd
{
    short data[IMG_CHANNEL][DCE_CHANNEL][3][3];
};
using qBConv3rd_t = struct qBConv3rd
{
    int data[IMG_CHANNEL];
};


/**
 * Initialize all data space
 */

extern RGBIOData_t* INDATA;
extern RGBIOData_t* dINDATA;

extern RGBIOData_t* OUTDATA;
extern RGBIOData_t* dOUTDATA;

extern qNormImg_t* dNORM;

extern qNetIO_t* NETIO;
extern qNetIO_t* dNETIO;

extern qNetFeature_t* dFEATURE1;
extern qNetFeature_t* dFEATURE2;

extern qEnhancedParam_t* PARAM;
extern qEnhancedParam_t* dPARAM;
extern qEnhancedParam_t* UPSBUFFER;
extern qEnhancedParam_t* dUPSBUFFER;

extern qWConv1st_t* CONVW01;
extern qBConv1st_t* CONVB01;
extern qWConv2nd_t* CONVW02;
extern qBConv2nd_t* CONVB02;
extern qWConv3rd_t* CONVW03;
extern qBConv3rd_t* CONVB03;

extern qWConv1st_t* dCONVW01;
extern qBConv1st_t* dCONVB01;
extern qWConv2nd_t* dCONVW02;
extern qBConv2nd_t* dCONVB02;
extern qWConv3rd_t* dCONVW03;
extern qBConv3rd_t* dCONVB03;

/**
 * GPU Kernel Functions
 */

__global__ void dNorm(RGBIOData_t* dINDATA, qNormImg_t* dNORM);
__global__ void dDownSample(qNormImg_t* dNORM, qNetIO_t* dNETIO);
__global__ void dConv1st(qNetIO_t* dNETIO, qWConv1st_t* dCONVW01, qBConv1st_t* dCONVB01, qNetFeature_t* dFEATURE1);
__global__ void dConv2nd(qNetFeature_t* dFEATURE1, qWConv2nd_t* dCONVW02, qBConv2nd_t* dCONVB02, qNetFeature_t* dFEATURE2);
__global__ void dConv3rd(qNetFeature_t* dFEATURE1, qNetFeature_t* dFEATURE2, qWConv3rd_t* dCONVW03, qBConv3rd_t* dCONVB03, qNetIO_t* dNETIO);
__global__ void dUpSample_x(qNetIO_t* dNETIO, qEnhancedParam_t* dUPSBUFFER);
__global__ void dUpSample_y(qEnhancedParam_t* dUPSBUFFER, qEnhancedParam_t* dPARAM);
__global__ void dEnhance(qNormImg_t* dNORM, qEnhancedParam_t* dPARAM, RGBIOData_t* dOUTDATA);
