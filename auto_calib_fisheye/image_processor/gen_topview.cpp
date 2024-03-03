#include <math.h>
#include <time.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <thread>
#include <vector>
#include "config_parser.h"
#include "defines.h"
#include "image_processor_context.h"
#include "image_processor_cuda.h"
#include "perception_config.pb.h"
#include "segment_image_processor_cuda.h"
#include "utils.h"

int main(int argc, char** argv)
{
    if (argc <= 3)
    {
        printf("Usage %s <dataset> <initial calibration> <image set dir> "
               "<output dir>\n",
               argv[0]);
        exit(-1);
    }

    std::string dataset = argv[1];

    std::string calib     = argv[2];
    std::string input     = argv[3];
    std::string output    = argv[4];
    std::string extension = ".png";
    // Mat imgb  = cv::imread(input + "/cam2" + extension);
    // Mat imgf  = cv::imread(input + "/cam1" + extension);
    // Mat imgl  = cv::imread(input + "/cam0" + extension);
    // Mat imgr  = cv::imread(input + "/cam3" + extension);
    Mat imgb = cv::imread(input + "/b" + extension);
    Mat imgf = cv::imread(input + "/f" + extension);
    Mat imgl = cv::imread(input + "/l" + extension);
    Mat imgr = cv::imread(input + "/r" + extension);

    using namespace perception::imgproc;
    auto config = std::make_shared<PerceptionConfig>();
    util::LoadProtoFromASCIIFile(
        "/home/kiennt63/dev/surround_cam_calib/auto_calib_fisheye/config/"
        "perception_config.textproto",
        config.get());
    auto imgprocContext = std::make_unique<ImageProcessorContext>(
        std::make_unique<ImageProcessorCuda>(config->imgproc_config()),
        std::make_unique<SegmentImageProcessorCuda>(config->imgproc_config()));

    if (!imgprocContext->init())
    {
        LOG_ERROR("Failed to initialize image processor context!");
        throw std::runtime_error("Cannot init attributes");
    }

    cv::Mat imgTop;
    imgprocContext->createTopViewImage(imgl, imgf, imgb, imgr, imgTop);
    cv::imwrite(output + "/topview_after.png", imgTop);

    return 0;
}
