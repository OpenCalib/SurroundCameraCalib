/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and
 * Research JSC. All rights reserved. All information contained here is
 * proprietary and confidential to VinAI. Any use, reproduction, or disclosure
 * without the written permission of VinAI is prohibited.
 */
#ifndef PARKING_PERCEPTION_IMAGE_PROCESSOR_CUDA_UTILS_H_
#define PARKING_PERCEPTION_IMAGE_PROCESSOR_CUDA_UTILS_H_

#include <string>

#include "logger/logger.h"

namespace perception {
namespace imgproc {
constexpr int NUM_CAMS = 4;

#define GET_LAST_CUDA_ERRORS()                         \
    {                                                  \
        cudaError_t err = cudaGetLastError();          \
        if (err != cudaSuccess)                        \
        {                                              \
            LOG_ERROR("{}: {}", cudaGetErrorName(err), \
                      cudaGetErrorString(err));        \
        }                                              \
    }

short* loadAndCopyMap(const std::string& fileName, int mapCols, int mapRows);
float* loadAndCopyMask(const std::string& fileName);
float* allocateMidtopview(size_t numPixels);
}  // namespace imgproc
}  // namespace perception

#endif  // PARKING_PERCEPTION_IMAGE_PROCESSOR_CUDA_UTILS_H_