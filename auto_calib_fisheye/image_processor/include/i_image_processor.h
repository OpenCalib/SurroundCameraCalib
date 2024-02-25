/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and
 * Research JSC. All rights reserved. All information contained here is
 * proprietary and confidential to VinAI. Any use, reproduction, or disclosure
 * without the written permission of VinAI is prohibited.
 */
#ifndef PARKING_PERCEPTION_IMAGE_PROCESSOR_I_IMAGE_PROCESSOR_H_
#define PARKING_PERCEPTION_IMAGE_PROCESSOR_I_IMAGE_PROCESSOR_H_

#include <opencv2/core/mat.hpp>

#include "proto/perception_config.pb.h"

using UVLists = std::array<std::vector<short>, 4>;

namespace perception {
namespace imgproc {
class IImageProcessor
{
public:
    IImageProcessor() = default;
    IImageProcessor(const ImageProcessorConfig& config) {}

    virtual bool init(const UVLists& uvLists)                               = 0;
    virtual void createTopViewImage(const cv::Mat& inImg0,
                                    const cv::Mat& inImg1,
                                    const cv::Mat& inImg2,
                                    const cv::Mat& inImg3, cv::Mat& outImg) = 0;
};
}  // namespace imgproc
}  // namespace perception

#endif  // PARKING_PERCEPTION_IMAGE_PROCESSOR_I_IMAGE_PROCESSOR_H_