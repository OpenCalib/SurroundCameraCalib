/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#ifndef PARKING_PERCEPTION_IMAGE_PROCESSOR_OPENGL_UTIL_H_
#define PARKING_PERCEPTION_IMAGE_PROCESSOR_OPENGL_UTIL_H_

#include <stdio.h>
#include <cstring>
#include <string>
#include <vector>

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <opencv2/opencv.hpp>

namespace perception {
namespace imgproc {

struct ImageData
{
    unsigned char *dataTex;
    unsigned int wTex;
    unsigned int hTex;
    GLint format;
};

struct SegmentImageData
{
    float *dataTex;
    unsigned int wTex;
    unsigned int hTex;
    GLint format;
};

void convertCVImageToImageData(const cv::Mat &srcImg, ImageData &destImg);

void convertSegmentImageToImageData(const cv::Mat &srcImg, SegmentImageData &destImg);

GLuint createTextureFromImage(ImageData *bmpData, GLuint textureID);

GLuint createTextureFromSegmentImage(SegmentImageData *bmpData, GLuint textureID);

GLuint buildShaderProgramFromFile(const char *vtxSrcPath, const char *pxlSrcPath, const char *name);

bool loadBowlObj(const char *str, std::vector<glm::vec3> &outVertices,
                 std::vector<glm::vec2> &outUVs);

bool readUVmapInformation(const std::string calibFilePath, std::vector<glm::vec2> &outUVs);

}  // namespace imgproc
}  // namespace perception

#endif  // PARKING_PERCEPTION_IMAGE_PROCESSOR_OPENGL_UTIL_H_