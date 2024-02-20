/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#ifndef PARKING_PERCEPTION_IMAGE_PROCESOR_SEGMENT_TOP_VIEW_STITCHING_H_
#define PARKING_PERCEPTION_IMAGE_PROCESOR_SEGMENT_TOP_VIEW_STITCHING_H_

#include <GL/glew.h>
#include <ctime>
#include <iostream>
#include <thread>
#include <vector>

#include "opengl_util.h"

namespace perception {
namespace imgproc {
#define MAX_SHAPE 4

class SegmentTopViewStitching
{
public:
    SegmentTopViewStitching();

    SegmentTopViewStitching(const std::string& fragShader, const std::string& vertShader,
                            const std::string& modelName, const std::string& textureTVImage0,
                            const std::string& textureTVImage1, const std::string& textureTVImage2,
                            const std::string& textureTVImage3, const std::string& calibInfo0,
                            const std::string& calibInfo1, const std::string& calibInfo2,
                            const std::string& calibInfo3);

    ~SegmentTopViewStitching();

    void loadResource();
    bool init(int width, int height);
    void deinit();
    bool render(int camPos);
    bool setTexture(GLuint tex, int camPos);
    bool createBlendingTextures(int camPos);
    bool createVBOs();

private:
    void loadUVsList();
    void loadBlendingData();
    void deleteTextures();
    void deleteVBOs();
    void draw(int camPos);

    int width_;
    int height_;

    ImageData* pImageData_[MAX_SHAPE];

    std::string fragShader_ = "./renderer/SegmentTopView.fsh";
    std::string vertShader_ = "./renderer/SegmentTopView.vsh";
    std::string modelName_  = "./data/calib/Bowl_topview_ver1_16_18.obj";

    std::string textureTVImageBlendings_[MAX_SHAPE] = {
        "./data/calib/alpha_TV_0.png", "./data/calib/alpha_TV_1.png", "./data/calib/alpha_TV_2.png",
        "./data/calib/alpha_TV_3.png"};
    std::string calibInfo_[MAX_SHAPE] = {
        "./data/calib/calib_cam0_topview.txt", "./data/calib/calib_cam1_topview.txt",
        "./data/calib/calib_cam2_topview.txt", "./data/calib/calib_cam3_topview.txt"};

    GLint programRenderPanorama_;
    GLuint textures_[MAX_SHAPE];
    GLuint textureBlendings_[MAX_SHAPE] = {0, 0, 0, 0};
    GLuint vertexBuffers_[MAX_SHAPE];
    GLuint uvBuffers_[MAX_SHAPE];
    cv::Mat alphaImg_[MAX_SHAPE];

    std::vector<glm::vec3> vertices_;
    std::vector<glm::vec2> uvs_;
    std::vector<glm::vec3> normals_;

    std::vector<std::vector<glm::vec2>> listOfUVs_;
    int verticesSize_;
    bool isInitSuccess_ = false;
};
}  // namespace imgproc
}  // namespace perception

#endif  // PARKING_PERCEPTION_IMAGE_PROCESOR_SEGMENT_TOP_VIEW_STITCHING_H_