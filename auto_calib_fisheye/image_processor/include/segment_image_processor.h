/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#ifndef PARKING_PERCEPTION_IMAGE_PROCESSOR_SEGMENT_IMAGE_PROCESSOR_H_
#define PARKING_PERCEPTION_IMAGE_PROCESSOR_SEGMENT_IMAGE_PROCESSOR_H_

#include <array>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "i_image_processor.h"
#include "segment_top_view_stitching.h"

namespace perception {
namespace imgproc {
class SegmentImageProcessor : public IImageProcessor
{
public:
    SegmentImageProcessor(const ImageProcessorConfig& config);
    ~SegmentImageProcessor();

    bool init() override;
    void createTopViewImage(const cv::Mat& inImg0, const cv::Mat& inImg1, const cv::Mat& inImg2,
                            const cv::Mat& inImg3, cv::Mat& outImg) override;
    void releaseResource();

private:
    void initTopViewStitching();
    int initRenderBuffer();
    void render();
    void stitchTopView(cv::Mat& img);

    SegmentTopViewStitching* pTopView_;
    GLuint textureId_;
    GLuint fboId_;
    std::array<GLuint, 4> texture_{0, 0, 0, 0};
};
}  // namespace imgproc
}  // namespace perception

#endif  // PARKING_PERCEPTION_IMAGE_PROCESSOR_SEGMENT_IMAGE_PROCESSOR_H_