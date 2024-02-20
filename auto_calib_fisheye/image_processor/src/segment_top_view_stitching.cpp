/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#include "segment_top_view_stitching.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "opengl_util.h"

#define SAFE_DELETE(a)           \
    if ((a) != NULL) delete (a); \
    (a) = NULL;

namespace perception {
namespace imgproc {

SegmentTopViewStitching::SegmentTopViewStitching()
{
    for (int i = 0; i < MAX_SHAPE; i++)
    {
        pImageData_[i] = new ImageData();
    }
}

SegmentTopViewStitching::SegmentTopViewStitching(
    const std::string& fragShader, const std::string& vertShader, const std::string& modelName,
    const std::string& textureTVImage0, const std::string& textureTVImage1,
    const std::string& textureTVImage2, const std::string& textureTVImage3,
    const std::string& calibInfo0, const std::string& calibInfo1, const std::string& calibInfo2,
    const std::string& calibInfo3)
    : fragShader_(fragShader),
      vertShader_(vertShader),
      modelName_(modelName),
      textureTVImageBlendings_{textureTVImage0, textureTVImage1, textureTVImage2, textureTVImage3},
      calibInfo_{calibInfo0, calibInfo1, calibInfo2, calibInfo3}
{
    for (int i = 0; i < MAX_SHAPE; i++)
    {
        pImageData_[i] = new ImageData();
    }
}

SegmentTopViewStitching::~SegmentTopViewStitching()
{
    for (int i = 0; i < MAX_SHAPE; i++)
    {
        free(pImageData_[i]->dataTex);
        SAFE_DELETE(pImageData_[i]);
    }
}

void SegmentTopViewStitching::loadResource()
{
    loadUVsList();
    loadBlendingData();
}

void SegmentTopViewStitching::loadBlendingData()
{
    for (int i = 0; i < MAX_SHAPE; i++)
    {
        alphaImg_[i] = cv::imread(textureTVImageBlendings_[i], cv::IMREAD_GRAYSCALE);
        convertCVImageToImageData(alphaImg_[i], *pImageData_[i]);
    }
}

bool SegmentTopViewStitching::init(int width, int height)
{
    width_  = width;
    height_ = height;

    programRenderPanorama_ = buildShaderProgramFromFile(vertShader_.c_str(), fragShader_.c_str(),
                                                        "programRenderPanorama");
    if (!programRenderPanorama_)
    {
        printf("Error build shader program");
        return false;
    }

    isInitSuccess_ = createVBOs();
    if (isInitSuccess_ == false)
    {
        printf("createVBO() returned error");
        return isInitSuccess_;
    }

    for (int i = 0; i < MAX_SHAPE; ++i)
    {
        createBlendingTextures(i);
    }

    return isInitSuccess_;
}

void SegmentTopViewStitching::loadUVsList()
{
    // Init data for bowl
    vertices_.clear();
    loadBowlObj(modelName_.c_str(), vertices_, uvs_);
    listOfUVs_.clear();
    for (int i = 0; i < MAX_SHAPE; ++i)
    {
        std::vector<glm::vec2> uvOuts;
        uvOuts.clear();
        if (readUVmapInformation(calibInfo_[i], uvOuts) == false)
        {
            printf("Cannot found calib information");
            break;
        }
        listOfUVs_.push_back(uvOuts);
    }
}

void SegmentTopViewStitching::deinit()
{
    deleteTextures();
    deleteVBOs();
}

void SegmentTopViewStitching::draw(int camPos)
{
    // 1rst attribute buffer : vertices
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vertexBuffers_[camPos]);
    glVertexAttribPointer(0,         // attribute
                          3,         // size
                          GL_FLOAT,  // type
                          GL_FALSE,  // normalized?
                          0,         // stride
                          (void*)0   // array buffer offset
    );

    // 2nd attribute buffer : UVs
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, uvBuffers_[camPos]);
    glVertexAttribPointer(1,         // attribute
                          2,         // size
                          GL_FLOAT,  // type
                          GL_FALSE,  // normalized?
                          0,         // stride
                          (void*)0   // array buffer offset
    );

    // Draw the triangle !
    glDrawArrays(GL_TRIANGLES, 0, verticesSize_);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
}

bool SegmentTopViewStitching::render(int camPos)
{
    if (!isInitSuccess_)
    {
        return false;
    }
    glUseProgram(programRenderPanorama_);
    glDisable(GL_CULL_FACE);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textures_[camPos]);
    glUniform1i(glGetUniformLocation(programRenderPanorama_, "sampler"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, textureBlendings_[camPos]);
    glUniform1i(glGetUniformLocation(programRenderPanorama_, "sampler_blend"), 1);

    glValidateProgram(programRenderPanorama_);

    draw(camPos);
    return true;
}

bool SegmentTopViewStitching::setTexture(GLuint tex, int camPos)
{
    textures_[camPos] = tex;

    return true;
}

bool SegmentTopViewStitching::createBlendingTextures(int camPos)
{
    textureBlendings_[camPos] =
        createTextureFromImage(pImageData_[camPos], textureBlendings_[camPos]);
    return true;
}

bool SegmentTopViewStitching::createVBOs()
{
    if (listOfUVs_.size() < MAX_SHAPE - 1)
    {
        printf("Cannot load calib - UV list");
        return false;
    }

    glGenBuffers(MAX_SHAPE, &vertexBuffers_[0]);
    glGenBuffers(MAX_SHAPE, &uvBuffers_[0]);

    for (int i = 0; i < MAX_SHAPE; ++i)
    {
        // Load it into a VBO
        glBindBuffer(GL_ARRAY_BUFFER, vertexBuffers_[i]);
        glBufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(glm::vec3), &vertices_[0],
                     GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, uvBuffers_[i]);
        glBufferData(GL_ARRAY_BUFFER, listOfUVs_[i].size() * sizeof(glm::vec2), &listOfUVs_[i][0],
                     GL_STATIC_DRAW);
    }

    verticesSize_ = vertices_.size();

    return true;
}

void SegmentTopViewStitching::deleteTextures()
{
    glDeleteTextures(MAX_SHAPE, &textures_[0]);
    glDeleteTextures(MAX_SHAPE, &textureBlendings_[0]);
    return;
}

void SegmentTopViewStitching::deleteVBOs()
{
    glDeleteBuffers(MAX_SHAPE, &vertexBuffers_[0]);
    glDeleteBuffers(MAX_SHAPE, &uvBuffers_[0]);

    return;
}
}  // namespace imgproc
}  // namespace perception