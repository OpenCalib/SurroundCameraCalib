/*
 * Copyright (c) 2020 - 2023, VINAI Artificial Intelligence Application and Research JSC.
 * All rights reserved. All information contained here is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */
#include <cuda_runtime_api.h>

#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace perception {
namespace imgproc {
short* loadAndCopyMap(const std::string& fileName, int mapCols, int mapRows)
{
    cv::Mat map(mapRows, mapCols, CV_16SC2);
    const int mapSize = map.total() * map.elemSize();
    std::ifstream file(fileName);
    {
        short x, y;
        char sep;
        size_t id         = 0;
        const size_t step = sizeof(short);
        while ((file >> x >> sep >> y) && (sep == ','))
        {
            std::memcpy(map.data + id, &x, step);
            std::memcpy(map.data + id + step, &y, step);
            id = id + 2 * step;
        }
    }
    file.close();
    short* mapPtr;
    cudaMalloc((void**)&mapPtr, mapSize);
    cudaMemcpy(mapPtr, map.data, mapSize, cudaMemcpyHostToDevice);
    return mapPtr;
}

float* loadAndCopyMask(const std::string& fileName)
{
    cv::Mat mask = cv::imread(fileName, cv::IMREAD_GRAYSCALE);
    mask.convertTo(mask, CV_32FC1, 1. / 255);
    const int maskSize = mask.total() * mask.elemSize();
    float* maskPtr;
    cudaMalloc((void**)&maskPtr, maskSize);
    cudaMemcpy(maskPtr, mask.data, maskSize, cudaMemcpyHostToDevice);
    return maskPtr;
}

float* allocateMidtopview(size_t numPixels)
{
    float* midtopPtr;
    const int midtopSize = numPixels * sizeof(float);
    cudaMalloc((void**)&midtopPtr, midtopSize);
    cudaMemset(midtopPtr, 0, midtopSize);
    return midtopPtr;
}
}  // namespace imgproc
}  // namespace perception