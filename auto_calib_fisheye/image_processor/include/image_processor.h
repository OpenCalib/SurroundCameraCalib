/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#ifndef PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_
#define PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_

#include <array>
#include <memory>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "i_image_processor.h"
#include "top_view_stitching.h"

namespace perception {
namespace imgproc {
class ImageProcessor : public IImageProcessor
{
public:
    ImageProcessor(const ImageProcessorConfig& config);
    ~ImageProcessor();

    bool init(const UVLists& uvLists) override;
    void createTopViewImage(const cv::Mat& in_img0, const cv::Mat& in_img1, const cv::Mat& in_img2,
                            const cv::Mat& in_img3, cv::Mat& out_img) override;
    void releaseResource();

private:
    TopViewStitching* pTopView_;
    GLuint textureId_;
    GLuint fboId_;
    std::array<GLuint, 4> texture_{0, 0, 0, 0};

    void initTopViewStitching();
    int initRenderBuffer();
    void render();
    void stitchTopView(cv::Mat& img);
};

GLFWwindow* initGL();
}  // namespace imgproc
}  // namespace perception

#endif  // PARKING_PERCEPTION_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_