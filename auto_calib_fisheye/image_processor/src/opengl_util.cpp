/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#include "opengl_util.h"

#include <stdio.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace perception {
namespace imgproc {
static GLuint loadShaderFromFile(GLenum type, const char *shaderPath, const char *name)
{
    bool isOpen;
    static_cast<void>(isOpen);
    GLint size;
    const char *data = NULL;

    FILE *pFile = fopen(shaderPath, "rb");
    if (pFile)
    {
        // Get the file size
        fseek(pFile, 0, SEEK_END);
        size = ftell(pFile);
        if (size < 0)
        {
            printf(" size is negative");
            fclose(pFile);
            return 0;
        }
        fseek(pFile, 0, SEEK_SET);

        // read the data, append a 0 byte as the data might represent a string
        char *pData   = new char[size + 1];
        pData[size]   = '\0';
        int bytesRead = fread(pData, 1, size, pFile);

        if (bytesRead != size)
        {
            delete[] pData;
            size = 0;
        }
        else
        {
            data   = pData;
            isOpen = true;
        }
        fclose(pFile);
    }
    else
    {
        size = 0;
    }

    // Create the shader object
    GLuint shader = glCreateShader(type);
    if (shader == 0)
    {
        delete[] data;
        return 0;
    }

    // Load and compile the shader
    glShaderSource(shader, 1, &data, nullptr);
    glCompileShader(shader);

    // Verify the compilation worked as expected
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled)
    {
        printf("Error compiling %s shader for %s\n", (type == GL_VERTEX_SHADER) ? "vtx" : "pxl",
               name);

        GLint size = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &size);
        if (size > 0)
        {
            // Get and report the error message
            std::unique_ptr<char> infoLog(new char[size]);
            glGetShaderInfoLog(shader, size, NULL, infoLog.get());
            printf("  msg:\n%s\n", infoLog.get());
        }

        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

// Create a program object given vertex and pixels shader source
GLuint buildShaderProgramFromFile(const char *vtxSrcPath, const char *pxlSrcPath, const char *name)
{
    GLuint program = glCreateProgram();
    if (program == 0)
    {
        printf("Failed to allocate program object\n");
        return 0;
    }

    // Compile the shaders and bind them to this program
    GLuint vertexShader = loadShaderFromFile(GL_VERTEX_SHADER, vtxSrcPath, name);
    if (vertexShader == 0)
    {
        printf("Failed to load vertex shader\n");
        glDeleteProgram(program);
        return 0;
    }
    GLuint pixelShader = loadShaderFromFile(GL_FRAGMENT_SHADER, pxlSrcPath, name);
    if (pixelShader == 0)
    {
        printf("Failed to load pixel shader\n");
        glDeleteProgram(program);
        glDeleteShader(vertexShader);
        return 0;
    }
    glAttachShader(program, vertexShader);
    glAttachShader(program, pixelShader);

    // Link the program
    glLinkProgram(program);
    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked)
    {
        printf("Error linking program.\n");
        GLint size = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &size);
        if (size > 0)
        {
            // Get and report the error message
            std::unique_ptr<char> infoLog(new char[size]);
            glGetProgramInfoLog(program, size, NULL, infoLog.get());
            printf("  msg:  %s\n", infoLog.get());
        }

        glDeleteProgram(program);
        glDeleteShader(vertexShader);
        glDeleteShader(pixelShader);
        return 0;
    }

#if 0  // Debug output to diagnose shader parameters
    GLint numShaderParams;
    GLchar paramName[128];
    GLint paramSize;
    GLenum paramType;
    const char *typeName = "?";
    printf("Shader parameters for %s:\n", name);
    glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &numShaderParams);
    for (GLint i=0; i<numShaderParams; i++) {
        glGetActiveUniform(program,
                           i,
                           sizeof(paramName),
                           nullptr,
                           &paramSize,
                           &paramType,
                           paramName);
        switch (paramType) {
            case GL_FLOAT:      typeName = "GL_FLOAT"; break;
            case GL_FLOAT_VEC4: typeName = "GL_FLOAT_VEC4"; break;
            case GL_FLOAT_MAT4: typeName = "GL_FLOAT_MAT4"; break;
            case GL_SAMPLER_2D: typeName = "GL_SAMPLER_2D"; break;
        }

        printf("  %2d: %s\t (%d) of type %s(%d)\n", i, paramName, paramSize, typeName, paramType);
    }
#endif

    return program;
}

void convertCVImageToImageData(const cv::Mat &srcImg, ImageData &destImg)
{
    destImg.dataTex = srcImg.data;
    destImg.wTex    = srcImg.cols;
    destImg.hTex    = srcImg.rows;
    switch (srcImg.channels())
    {
        case 3:
            destImg.format = GL_RGB;
            break;
        case 4:
            destImg.format = GL_RGBA;
            break;
        case 1:
            destImg.format = GL_RED;
            break;
        default:
            break;
    }
}

void convertSegmentImageToImageData(const cv::Mat &srcImg, SegmentImageData &destImg)
{
    destImg.dataTex = reinterpret_cast<float *>(srcImg.data);
    destImg.wTex    = srcImg.cols;
    destImg.hTex    = srcImg.rows;
    switch (srcImg.channels())
    {
        case 3:
            destImg.format = GL_RGB;
            break;
        case 4:
            destImg.format = GL_RGBA;
            break;
        case 1:
            destImg.format = GL_RED;
            break;
        default:
            break;
    }
}

GLuint createTextureFromImage(ImageData *bmpData, GLuint textureID)
{
    // Create one OpenGL texture
    if (textureID == 0)
    {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        // Give the image to OpenGL
        glTexImage2D(GL_TEXTURE_2D, 0, bmpData->format, bmpData->wTex, bmpData->hTex, 0,
                     bmpData->format, GL_UNSIGNED_BYTE, bmpData->dataTex);

        GLint err = glGetError();
        if (err != GL_NO_ERROR)
        {
            printf("glTexImage2D return ERROR: %d", err);
        }

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bmpData->wTex, bmpData->hTex, bmpData->format,
                        GL_UNSIGNED_BYTE, bmpData->dataTex);
    }

    glGenerateMipmap(GL_TEXTURE_2D);

    return textureID;
}

GLuint createTextureFromSegmentImage(SegmentImageData *bmpData, GLuint textureID)
{
    // Create one OpenGL texture
    if (textureID == 0)
    {
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        // Give the image to OpenGL
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, bmpData->wTex, bmpData->hTex, 0, GL_RGB, GL_FLOAT,
                     bmpData->dataTex);

        GLint err = glGetError();
        if (err != GL_NO_ERROR)
        {
            printf("glTexImage2D return ERROR: %d", err);
        }

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, bmpData->wTex, bmpData->hTex, GL_RGB, GL_FLOAT,
                        bmpData->dataTex);
    }

    glGenerateMipmap(GL_TEXTURE_2D);

    return textureID;
}

bool loadBowlObj(const char *path, std::vector<glm::vec3> &outVertices,
                 std::vector<glm::vec2> &outUVs)
{
    printf("Loading OBJ file %s...\n", path);

    std::vector<unsigned int> vertexIndices, uvIndices, normalIndices;
    std::vector<glm::vec3> tempVertices;
    std::vector<glm::vec2> tempUVs;

    FILE *file = fopen(path, "r");
    if (file == NULL)
    {
        printf("Impossible to open the file ! Are you in the right path ? See "
               "Tutorial 1 for details\n");
        return false;
    }

    char splitToken[] = " /\n";

    int size   = 100;
    size_t len = size;
    char *line = (char *)malloc(sizeof(char) * size);
    ssize_t read;
    while ((read = getline(&line, &len, file)) != -1)
    {
        char *next = NULL;
        char *value;
        value = strtok_r(line, splitToken, &next);
        if (strcmp(value, "v") == 0)
        {
            glm::vec3 vertex;

            value = strtok_r(NULL, splitToken, &next);
            if (value != NULL)
            {
                vertex.x = atof(value);
            }
            else
            {
                return false;
            }

            value = strtok_r(NULL, splitToken, &next);
            if (value != NULL)
            {
                vertex.y = atof(value);
            }
            else
            {
                return false;
            }

            value = strtok_r(NULL, splitToken, &next);
            if (value != NULL)
            {
                vertex.z = atof(value);
            }
            else
            {
                return false;
            }

            tempVertices.push_back(vertex);
        }
        else if (strcmp(value, "vt") == 0)
        {
            glm::vec2 uv;

            value = strtok_r(NULL, splitToken, &next);
            if (value != NULL)
            {
                uv.x = atof(value);
            }
            else
            {
                return false;
            }

            value = strtok_r(NULL, splitToken, &next);
            if (value != NULL)
            {
                uv.y = atof(value);
            }
            else
            {
                return false;
            }

            uv.y = -uv.y;  // Invert V coordinate since we will only use DDS
                           // texture, which are inverted. Remove if you want to
                           // use TGA or BMP loaders.
            tempUVs.push_back(uv);
        }
        else if (strcmp(value, "f") == 0)
        {
            int indexSize = 3;
            unsigned int vertexIndex[indexSize], uvIndex[indexSize], normalIndex[indexSize];

            for (int i = 0; i < indexSize; i++)
            {
                value = strtok_r(NULL, splitToken, &next);
                if (value != NULL)
                {
                    vertexIndex[i] = atoi(value);
                }
                else
                {
                    return false;
                }

                value = strtok_r(NULL, splitToken, &next);
                if (value != NULL)
                {
                    uvIndex[i] = atoi(value);
                }
                else
                {
                    return false;
                }

                value = strtok_r(NULL, splitToken, &next);
                if (value != NULL)
                {
                    normalIndex[i] = atoi(value);
                }
                else
                {
                    return false;
                }

                vertexIndices.push_back(vertexIndex[i]);
                uvIndices.push_back(uvIndex[i]);
                normalIndices.push_back(normalIndex[i]);
            }
        }
    }
    free(line);
    fclose(file);

    // For each vertex of each triangle
    for (unsigned int i = 0; i < vertexIndices.size(); i++)
    {
        // Get the indices of its attributes
        unsigned int vertexIndex = vertexIndices[i];
        unsigned int uvIndex     = uvIndices[i];

        // Get the attributes thanks to the index
        glm::vec3 vertex = tempVertices[vertexIndex - 1];
        glm::vec2 uv     = tempUVs[uvIndex - 1];

        // Put the attributes in buffers
        outVertices.push_back(vertex);
        outUVs.push_back(uv);
    }
    return true;
}

int readSize      = 30;
char splitToken[] = " \n";

bool readUVmapInformation(const std::string calibFilePath, std::vector<glm::vec2> &outUVs)
{
    FILE *file = fopen(calibFilePath.c_str(), "r");
    if (file == NULL)
    {
        printf("Cannot open file %s", calibFilePath.c_str());
        return false;
    }

    glm::vec2 uv;

    size_t len = readSize;
    char *line = (char *)malloc(sizeof(char) * readSize);
    ssize_t read;

    while ((read = getline(&line, &len, file)) != -1)
    {
        char *next = NULL;
        char *value;
        int count;
        value = strtok_r(line, splitToken, &next);
        if (value != NULL)
        {
            uv.x = atof(value);
        }
        else
        {
            return false;
        }

        value = strtok_r(NULL, splitToken, &next);
        if (value != NULL)
        {
            uv.y = atof(value);
        }
        else
        {
            return false;
        }

        value = strtok_r(NULL, splitToken, &next);
        if (value != NULL)
        {
            count = atoi(value);
        }
        else
        {
            return false;
        }

        for (int i = 0; i < count; i++)
        {
            outUVs.push_back(uv);
        }
    }

    free(line);
    fclose(file);
    return true;
}

}  // namespace imgproc
}  // namespace perception