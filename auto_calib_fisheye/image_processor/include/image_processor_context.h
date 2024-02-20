/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#ifndef PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_CONTEXT_H_
#define PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_CONTEXT_H_

#include <memory>

#include "i_image_processor.h"

namespace perception {
namespace imgproc {
class ImageProcessorContext
{
public:
    ImageProcessorContext() = default;
    ImageProcessorContext(std::unique_ptr<IImageProcessor>&& imgProcessor,
                          std::unique_ptr<IImageProcessor>&& segImgProcessor);

    bool init();
    void createTopViewImage(const cv::Mat& inImg0, const cv::Mat& inImg1, const cv::Mat& inImg2,
                            const cv::Mat& inImg3, cv::Mat& outImg);
    void createSegTopViewImage(const cv::Mat& inImg0, const cv::Mat& inImg1, const cv::Mat& inImg2,
                               const cv::Mat& inImg3, cv::Mat& outImg);

private:
    std::unique_ptr<IImageProcessor> imgProcessor_;
    std::unique_ptr<IImageProcessor> segImgProcessor_;
};
}  // namespace imgproc
}  // namespace perception

#endif  // PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_CONTEXT_H_