#include "main.cuh"
#include <chrono>

int main(int argc, char** argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);

#ifdef ON_JETSON
    int capWidth = IMG_WIDTH;
    int capHeight = IMG_HIGHT;
    int disWidth = IMG_WIDTH;
    int disHeight = IMG_HIGHT;
    int framerate = 30;
    int flip_method = 0;

    std::string pipeline = cvf::gstreamer_pipeline(capWidth,
	capHeight,
	disWidth,
	disHeight,
	framerate,
	flip_method);

    printf("Using Pipeline : %s\n", pipeline.c_str());

    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        cap.release();
        perror("Failed to open camera\n");
        exit(EXIT_FAILURE);
    }

    cv::Mat capImg;

    if (!cap.read(capImg)) {
        perror("Captrue read error\n");
        cap.release();
        exit(EXIT_FAILURE);
    }

    cv::imwrite("testInput.png", capImg);

    cap.release();
#endif

    auto start = std::chrono::steady_clock::now();
    DCE_ZeroCopy::initMem_ZeroCopy();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("initMem Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    loadWeight();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("loadWeight Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    cvf::cvReadImg("testInput.png");
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("cvReadImg Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE_ZeroCopy::qNormNDownSample_ZeroCopy();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qNormNDownSample Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE_ZeroCopy::qConv1st_ZeroCopy();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qConv1st Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE_ZeroCopy::qConv2nd_ZeroCopy();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qConv2nd Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE_ZeroCopy::qConv3rd_ZeroCopy();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qConv3rd Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE_ZeroCopy::qUpSample_ZeroCopy();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qUpSample Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    DCE_ZeroCopy::qEnhance_ZeroCopy();
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("qEnhance Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    start = std::chrono::steady_clock::now();
    cvf::cvOutputImg("Enhanced_CPP_output.png");
    end = std::chrono::steady_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("cvOutputImg Function Time: %lf\n", static_cast<double>(duration.count()) / 1000.0);
    printf("\n\n------------------------------------------------------------\n\n");

    DCE_ZeroCopy::cleanMem_ZeroCopy();

    return 0;
}
