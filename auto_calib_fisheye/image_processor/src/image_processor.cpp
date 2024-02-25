/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#include "image_processor.h"

#include "opengl_util.h"

namespace perception {
namespace imgproc {
constexpr int WIDTH  = 800;
constexpr int HEIGHT = 900;

ImageProcessor::ImageProcessor(const ImageProcessorConfig& config) : IImageProcessor(config)
{
    const std::string fragShader      = config.frag_shader();
    const std::string vertShader      = config.vert_shader();
    const std::string modelName       = config.model_name();
    const std::string textureTVImage0 = config.texture_tv_image_0();
    const std::string textureTVImage1 = config.texture_tv_image_1();
    const std::string textureTVImage2 = config.texture_tv_image_2();
    const std::string textureTVImage3 = config.texture_tv_image_3();
    const std::string calibInfo0      = config.calib_info_0();
    const std::string calibInfo1      = config.calib_info_1();
    const std::string calibInfo2      = config.calib_info_2();
    const std::string calibInfo3      = config.calib_info_3();
    pTopView_ = new TopViewStitching(fragShader, vertShader, modelName, textureTVImage0,
                                     textureTVImage1, textureTVImage2, textureTVImage3, calibInfo0,
                                     calibInfo1, calibInfo2, calibInfo3);
}

ImageProcessor::~ImageProcessor()
{
    releaseResource();
}

bool ImageProcessor::init(const UVLists& uvLists)
{
    initTopViewStitching();

    if (initRenderBuffer() == -1)
    {
        std::cerr << "Failed to initialize render buffer!" << std::endl;
        return false;
    }

    return true;
}

void ImageProcessor::createTopViewImage(const cv::Mat& inImg0, const cv::Mat& inImg1,
                                        const cv::Mat& inImg2, const cv::Mat& inImg3,
                                        cv::Mat& outImg)
{
    std::array<ImageData, 4> imageData;
    convertCVImageToImageData(inImg0, imageData[0]);
    convertCVImageToImageData(inImg1, imageData[1]);
    convertCVImageToImageData(inImg2, imageData[2]);
    convertCVImageToImageData(inImg3, imageData[3]);
    for (int i = 0; i < 4; i++)
    {
        texture_[i] = createTextureFromImage(&imageData[i], texture_[i]);
    }
    for (int i = 0; i < 4; i++)
    {
        pTopView_->setTexture(texture_[i], i);
    }

    stitchTopView(outImg);
}

void ImageProcessor::initTopViewStitching()
{
    pTopView_->loadResource();
    pTopView_->init(WIDTH, HEIGHT);
}

int ImageProcessor::initRenderBuffer()
{
    glGenTextures(1, &textureId_);
    glBindTexture(GL_TEXTURE_2D, textureId_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP,
                    GL_TRUE);  // automatic mipmap
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // create a framebuffer object
    GLuint fboId;
    glGenFramebuffers(1, &fboId);
    glBindFramebuffer(GL_FRAMEBUFFER, fboId);

    // attach the texture to FBO color attachment point
    glFramebufferTexture2D(GL_FRAMEBUFFER,        // 1. fbo target: GL_FRAMEBUFFER
                           GL_COLOR_ATTACHMENT0,  // 2. attachment point
                           GL_TEXTURE_2D,         // 3. tex target: GL_TEXTURE_2D
                           textureId_,            // 4. tex ID
                           0);                    // 5. mipmap level: 0(base)

    // check FBO status
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE)
    {
        return -1;
    }

    // switch back to window-system-provided framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    return 0;
}

void ImageProcessor::render()
{
    glViewport(0, 0, WIDTH, HEIGHT);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pTopView_->render(0);
    pTopView_->render(3);
    pTopView_->render(1);
    pTopView_->render(2);
}

void ImageProcessor::stitchTopView(cv::Mat& img)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fboId_);

    render();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, textureId_);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    img = cv::Mat(HEIGHT, WIDTH, CV_8UC4);

    glBindFramebuffer(GL_FRAMEBUFFER, fboId_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureId_, 0);
    glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, img.data);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void ImageProcessor::releaseResource()
{
    if (pTopView_)
    {
        pTopView_->deinit();
    }
}

GLFWwindow* initGL()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        return nullptr;
    }

    // glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);  // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                   GL_TRUE);  // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE,
                   GLFW_OPENGL_CORE_PROFILE);  // We don't want the old
    glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

    // Open a window and create its OpenGL context
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "", nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "Failed to open GLFW window. If you have an Intel GPU, they are "
                     "not 3.3 compatible. Try the 2.1 version of the tutorials."
                  << std::endl;
        glfwTerminate();
        return nullptr;
    }
    glfwMakeContextCurrent(window);  // Initialize GLEW
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW." << std::endl;
        return nullptr;
    }

    float points[] = {1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f};

    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 12 * sizeof(float), points, GL_STATIC_DRAW);
    GLuint vao = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    return nullptr;
}
}  // namespace imgproc
}  // namespace perception