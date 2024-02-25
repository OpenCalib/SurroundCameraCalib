/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#ifndef PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_CUDA_H_
#define PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_CUDA_H_

#include "cuda_utils.h"
#include "i_image_processor.h"

namespace perception {
namespace imgproc {
class ImageProcessorCuda : public IImageProcessor
{
public:
    ImageProcessorCuda(const ImageProcessorConfig& config);
    ~ImageProcessorCuda();

    bool init(const UVLists& uvlists) override;
    void createTopViewImage(const cv::Mat& fisheye0, const cv::Mat& fisheye1,
                            const cv::Mat& fisheye2, const cv::Mat& fisheye3,
                            cv::Mat& topImg) override;

private:
    unsigned char* allocateFisheye();

    short* mapPtr[NUM_CAMS];
    float* maskPtr[NUM_CAMS];
    unsigned char* fisheyePtr[NUM_CAMS];
    float* midtopPtr[NUM_CAMS];

    std::string calibDir;
};
}  // namespace imgproc
}  // namespace perception

#endif  // PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_CUDA_H_