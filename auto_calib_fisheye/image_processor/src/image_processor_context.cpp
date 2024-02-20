/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#include "image_processor_context.h"

#include "logger/logger.h"

namespace perception {
namespace imgproc {
ImageProcessorContext::ImageProcessorContext(std::unique_ptr<IImageProcessor>&& imgProcessor,
                                             std::unique_ptr<IImageProcessor>&& segImgProcessor)
    : imgProcessor_(std::move(imgProcessor)),
      segImgProcessor_(std::move(segImgProcessor))
{
}

bool ImageProcessorContext::init()
{
    if (!imgProcessor_->init())
    {
        LOG_ERROR("Failed to initialize image processor!");
        return false;
    }
    if (!segImgProcessor_->init())
    {
        LOG_ERROR("Failed to initialize segmentation image processor!");
        return false;
    }
    return true;
}

void ImageProcessorContext::createTopViewImage(const cv::Mat& inImg0, const cv::Mat& inImg1,
                                               const cv::Mat& inImg2, const cv::Mat& inImg3,
                                               cv::Mat& outImg)
{
    imgProcessor_->createTopViewImage(inImg0, inImg1, inImg2, inImg3, outImg);
}

void ImageProcessorContext::createSegTopViewImage(const cv::Mat& inImg0, const cv::Mat& inImg1,
                                                  const cv::Mat& inImg2, const cv::Mat& inImg3,
                                                  cv::Mat& outImg)
{
    segImgProcessor_->createTopViewImage(inImg0, inImg1, inImg2, inImg3, outImg);
}
}  // namespace imgproc
}  // namespace perception