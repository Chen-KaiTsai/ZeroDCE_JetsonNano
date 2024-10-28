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
qEnhancedParam_t* dUPSBUFFER = nullptr;
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

void loadWeight() {
    memcpy((void*)CONVW01, conv1_w, 2 * 864);
    memcpy((void*)CONVB01, conv1_b, 4 * 32);
    memcpy((void*)CONVW02, conv2_w, 2 * 9216);
    memcpy((void*)CONVB02, conv2_b, 4 * 32);
    memcpy((void*)CONVW03, conv3_w, 2 * 864);
    memcpy((void*)CONVB03, conv3_b, 4 * 3);
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
