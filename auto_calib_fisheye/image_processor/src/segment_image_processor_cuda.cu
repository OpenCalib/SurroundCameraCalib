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

#include "segment_image_processor_cuda.h"

namespace perception {
namespace imgproc {
constexpr int FISHEYE_COLS     = 320;
constexpr int FISHEYE_ROWS     = 320;
constexpr int FISHEYE_CHANNELS = 3;
constexpr int TOP_COLS         = 160;
constexpr int TOP_ROWS         = 180;
constexpr int TOP_CHANNELS     = 1;
constexpr int MAP_COLS         = TOP_COLS;
constexpr int MAP_ROWS         = TOP_ROWS;
constexpr int MAP_CHANNELS     = 2;
constexpr int MASK_COLS        = TOP_COLS;
constexpr int MASK_ROWS        = TOP_ROWS;
constexpr int MASK_CHANNELS    = 1;

constexpr int BLOCKS_PER_DIM_X    = 5;
constexpr int BLOCKS_PER_DIM_Y    = 6;
constexpr int THREADS_PER_BLOCK_X = 32;
constexpr int THREADS_PER_BLOCK_Y = 32;
constexpr int COLS_PER_THREAD     = 1;
constexpr int ROWS_PER_THREAD     = 1;

static __global__ void remapAndMaskImage(const float* __restrict fisheyePtr,
                                         const short* __restrict mapPtr,
                                         const float* __restrict maskPtr,
                                         float* __restrict midtopPtr)
{
    const int y0 = ROWS_PER_THREAD * (THREADS_PER_BLOCK_Y * blockIdx.y + threadIdx.y);
    int y1       = ROWS_PER_THREAD * (THREADS_PER_BLOCK_Y * blockIdx.y + threadIdx.y + 1);
    const int x0 = COLS_PER_THREAD * (THREADS_PER_BLOCK_X * blockIdx.x + threadIdx.x);
    int x1       = COLS_PER_THREAD * (THREADS_PER_BLOCK_X * blockIdx.x + threadIdx.x + 1);

    x1 = (x1 < TOP_COLS) ? x1 : TOP_COLS;
    y1 = (y1 < TOP_ROWS) ? y1 : TOP_ROWS;
    for (int y = y0; y < y1; y++)
    {
        for (int x = x0; x < x1; x++)
        {
            const int mapID    = MAP_CHANNELS * (y * MAP_COLS + x);
            const short coordX = mapPtr[mapID + 0];
            const short coordY = mapPtr[mapID + 1];
            if ((coordX < 0) || (coordX > FISHEYE_COLS - 1) || (coordY < 0) ||
                (coordY > FISHEYE_ROWS - 1))
            {
                continue;
            }

            const int fisheyeID = FISHEYE_CHANNELS * (coordY * FISHEYE_COLS + coordX);
            const int topID     = TOP_CHANNELS * (y * TOP_COLS + x);
            const int maskID    = MASK_CHANNELS * (y * MASK_COLS + x);
            const auto intensB  = fisheyePtr[fisheyeID + 0];
            const auto intensG  = fisheyePtr[fisheyeID + 1];
            const auto intensR  = fisheyePtr[fisheyeID + 2];
            float intens        = 0.f;
            if ((intensG > intensR) && (intensG > intensB))
            {
                intens = intensG;
            }
            else if (intensR > intensB)
            {
                intens = 1.f - intensR;
            }
            else
            {
                intens = 1.f - intensB;
            }
            const auto alpha = maskPtr[maskID];
            midtopPtr[topID] = intens * alpha;
        }
    }
}

static __global__ void combineImagePortion(const float* __restrict midtop0Ptr,
                                           const float* __restrict midtop1Ptr,
                                           const float* __restrict midtop2Ptr,
                                           const float* __restrict midtop3Ptr,
                                           float* __restrict topPtr)
{
    const int y0 = ROWS_PER_THREAD * (THREADS_PER_BLOCK_Y * blockIdx.y + threadIdx.y);
    int y1       = ROWS_PER_THREAD * (THREADS_PER_BLOCK_Y * blockIdx.y + threadIdx.y + 1);
    const int x0 = COLS_PER_THREAD * (THREADS_PER_BLOCK_X * blockIdx.x + threadIdx.x);
    int x1       = COLS_PER_THREAD * (THREADS_PER_BLOCK_X * blockIdx.x + threadIdx.x + 1);

    x1 = (x1 < TOP_COLS) ? x1 : TOP_COLS;
    y1 = (y1 < TOP_ROWS) ? y1 : TOP_ROWS;
    for (int y = y0; y < y1; y++)
    {
        for (int x = x0; x < x1; x++)
        {
            const int id = TOP_CHANNELS * (y * TOP_COLS + x);
            for (int z = 0; z < TOP_CHANNELS; z++)
            {
                topPtr[id + z] = midtop0Ptr[id + z] + midtop1Ptr[id + z] + midtop2Ptr[id + z] +
                                 midtop3Ptr[id + z];
            }
        }
    }
}

SegmentImageProcessorCuda::SegmentImageProcessorCuda(const ImageProcessorConfig& config)
    : IImageProcessor(config)
{
    calibDir = config.calib_dir();
}

SegmentImageProcessorCuda::~SegmentImageProcessorCuda()
{
    for (int id = 0; id < NUM_CAMS; id++)
    {
        cudaFree(mapPtr[id]);
        cudaFree(maskPtr[id]);
        cudaFreeHost(fisheyePtr[id]);
        cudaFree(midtopPtr[id]);
    }
}

bool SegmentImageProcessorCuda::init(const UVLists& uvLists)
{
    for (int id = 0; id < NUM_CAMS; id++)
    {
        // Load and copy maps and masks to device memory
        mapPtr[id]  = loadAndCopyMap(calibDir + "topview_seg/map" + std::to_string(id) + ".txt",
                                    MAP_COLS, MAP_ROWS);
        maskPtr[id] = loadAndCopyMask(calibDir + "topview_seg/mask" + std::to_string(id) + ".png");

        // Allocate pinned memory for fisheye images
        fisheyePtr[id] = allocateFisheye();

        // Allocate device memory to mid-topviews
        midtopPtr[id] = allocateMidtopview(TOP_ROWS * TOP_COLS * TOP_CHANNELS);
    }

    return true;
}

void SegmentImageProcessorCuda::createTopViewImage(const cv::Mat& fisheye0, const cv::Mat& fisheye1,
                                                   const cv::Mat& fisheye2, const cv::Mat& fisheye3,
                                                   cv::Mat& topImg)
{
    // Allocate device memory to topview
    topImg = cv::Mat(TOP_ROWS, TOP_COLS, CV_32FC1);
    float* topPtr;
    const int topSize = topImg.total() * topImg.elemSize();
    cudaMalloc((void**)&topPtr, topSize);
    GET_LAST_CUDA_ERRORS();

    // Perform topview generation: remapping & masking
    cudaStream_t topStream[NUM_CAMS];
    for (int id = 0; id < NUM_CAMS; id++)
    {
        cudaStreamCreate(&topStream[id]);
    }
    const cv::Mat fisheye[NUM_CAMS] = {fisheye0, fisheye1, fisheye2, fisheye3};
    for (int id = 0; id < NUM_CAMS; id++)
    {
        cudaMemcpyAsync((void*)fisheyePtr[id], fisheye[id].data,
                        fisheye[id].total() * fisheye[id].elemSize(), cudaMemcpyHostToDevice,
                        topStream[id]);
        remapAndMaskImage<<<dim3{BLOCKS_PER_DIM_X, BLOCKS_PER_DIM_Y, 1},
                            dim3{THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1}, 0, topStream[id]>>>(
            fisheyePtr[id], mapPtr[id], maskPtr[id], midtopPtr[id]);
    }
    cudaDeviceSynchronize();
    for (int id = 0; id < NUM_CAMS; id++)
    {
        cudaStreamDestroy(topStream[id]);
    }
    GET_LAST_CUDA_ERRORS();

    // Perform topview generation: combining
    combineImagePortion<<<dim3{BLOCKS_PER_DIM_X, BLOCKS_PER_DIM_Y, 1},
                          dim3{THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, 1}>>>(
        midtopPtr[0], midtopPtr[1], midtopPtr[2], midtopPtr[3], topPtr);
    cudaMemcpy(topImg.data, topPtr, topImg.total() * topImg.elemSize(), cudaMemcpyDeviceToHost);
    GET_LAST_CUDA_ERRORS();

    // Free resources
    cudaFree(topPtr);
    GET_LAST_CUDA_ERRORS();
}

float* SegmentImageProcessorCuda::allocateFisheye()
{
    float* fisheyePtr;
    constexpr int fisheyeSize = FISHEYE_COLS * FISHEYE_ROWS * FISHEYE_CHANNELS * sizeof(float);
    cudaMallocHost((void**)&fisheyePtr, fisheyeSize);
    return fisheyePtr;
}
}  // namespace imgproc
}  // namespace perception