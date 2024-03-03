#include "generate_lut.h"
#include <fstream>
#include "logger/logger.h"

#define IMG_HEIGHT 800
#define IMG_WIDTH  1280
#define TOPVIEW_H  18
#define TOPVIEW_W  16

void lut::genLUT(const Eigen::Matrix3d& matR, const Eigen::Vector3d& vecT,
                 const Eigen::Matrix3d& matK, const std::vector<double>& matD,
                 const std::string& lutOutputFile)
{
    std::ofstream outputFile(lutOutputFile);

    if (!outputFile.is_open())
    {
        LOG_ERROR("Cannot open file: {}", lutOutputFile);
    }

    for (size_t i = 0; i < TOPVIEW_H * 10; ++i)
    {
        for (size_t j = 0; j < TOPVIEW_W * 10; ++j)
        {
            // float xWorld = j / 10.0 - TOPVIEW_W / 2.0;
            // float yWorld = TOPVIEW_H / 2.0 - i / 10.0;

            float x_w = TOPVIEW_H / 2.0 - i / 10.0;
            float y_w = TOPVIEW_W / 2.0 - j / 10.0;

            std::vector<float> vecWorldPosRotate = {x_w, y_w, 0};
            float vecPointTransformed[3];
            vecPointTransformed[0] = matR(0, 0) * vecWorldPosRotate[0] +
                                     matR(0, 1) * vecWorldPosRotate[1] +
                                     matR(0, 2) * vecWorldPosRotate[2] + vecT[0];
            vecPointTransformed[1] = matR(1, 0) * vecWorldPosRotate[0] +
                                     matR(1, 1) * vecWorldPosRotate[1] +
                                     matR(1, 2) * vecWorldPosRotate[2] + vecT[1];
            vecPointTransformed[2] = matR(2, 0) * vecWorldPosRotate[0] +
                                     matR(2, 1) * vecWorldPosRotate[1] +
                                     matR(2, 2) * vecWorldPosRotate[2] + vecT[2];
            float lenVec3D = sqrt(vecPointTransformed[0] * vecPointTransformed[0] +
                                  vecPointTransformed[1] * vecPointTransformed[1] +
                                  vecPointTransformed[2] * vecPointTransformed[2]);
            float lenVec2D = sqrt(vecPointTransformed[0] * vecPointTransformed[0] +
                                  vecPointTransformed[1] * vecPointTransformed[1]);
            float ratio    = 0.0;
            float xd, yd;
            std::vector<int> vecPoint2D(2, 0.0);
            float theta = 0.0f;
            if (lenVec3D > 1e-6f)
            {
                ratio = lenVec2D / lenVec3D;
                theta = asin(ratio);
                if (vecPointTransformed[2] < 0) theta = M_PI - theta;
                float thetaSq = theta * theta;
                float cdist   = theta * (1.0 + matD[0] * thetaSq + matD[1] * thetaSq * thetaSq +
                                       matD[2] * thetaSq * thetaSq * thetaSq +
                                       matD[3] * thetaSq * thetaSq * thetaSq * thetaSq);
                xd            = vecPointTransformed[0] / lenVec2D * cdist;
                yd            = vecPointTransformed[1] / lenVec2D * cdist;
                vecPoint2D[0] = int(matK(0, 0) * xd + matK(0, 2));
                vecPoint2D[1] = int(matK(1, 1) * yd + matK(1, 2));
            }
            else
            {
                vecPoint2D[0] = int(matK(0, 2));
                vecPoint2D[1] = int(matK(1, 2));
            }
            if ((vecPoint2D[1] >= 0.0 && vecPoint2D[1] <= IMG_HEIGHT) &&
                (vecPoint2D[0] >= 0.0 && vecPoint2D[0] <= IMG_WIDTH))
            {
                // listUVs.at(y * TOPVIEW_W * 10 + x) =
                // std::make_pair(int(vecPoint2D[0]), int(vecPoint2D[1]));
                outputFile << int(vecPoint2D[0]) << " " << int(vecPoint2D[1]) << "\n";
            }
            else
            {
                // listUVs.at(y * TOPVIEW_W * 10 + x) =
                // std::make_pair(-2000, -2000);
                outputFile << -2000 << " " << -2000 << "\n";
            }
        }
    }
    outputFile.close();
}

void lut::genLUT(CamID camPos, const Eigen::Matrix3d& matR, const Eigen::Vector3d& vecT,
                 const Eigen::Matrix3d& matK, const std::vector<double>& matD,
                 std::vector<short>& uvList, const std::string& lutOutputFile, bool isSeg)
{
    std::ofstream outputFile(lutOutputFile);

    if (!outputFile.is_open())
    {
        LOG_ERROR("Cannot open file: {}", lutOutputFile);
    }

    constexpr float W      = 1280.0;
    constexpr float H      = 800.0;
    constexpr float PLGR_W = 8;
    constexpr float PLGR_H = 9;

    float IMG_W = 1280;
    float IMG_H = 800;
    int LUT_W   = 800;
    int LUT_H   = 900;

    if (isSeg)
    {
        IMG_W = 320;
        IMG_H = 320;
        LUT_W = 160;
        LUT_H = 180;
    }

    if (uvList.size() != LUT_W * LUT_H * 2)
    {
        LOG_ERROR("uvList.size() not matching actual size. {} vs {}", uvList.size(),
                  LUT_W * LUT_H * 2);
    }

    for (int i = 0; i < LUT_H; i++)
    {
        for (int j = 0; j < LUT_W; j++)
        {
            std::vector<float> vecWorldPosRotate;
            vecWorldPosRotate.push_back(PLGR_H - float(i) / LUT_H * PLGR_H * 2);
            vecWorldPosRotate.push_back(PLGR_W - float(j) / LUT_W * PLGR_W * 2);
            vecWorldPosRotate.push_back(0);
            std::vector<float> vecPointTransformed(3);
            vecPointTransformed[0] = matR(0, 0) * vecWorldPosRotate[0] +
                                     matR(0, 1) * vecWorldPosRotate[1] +
                                     matR(0, 2) * vecWorldPosRotate[2] + vecT[0];
            vecPointTransformed[1] = matR(1, 0) * vecWorldPosRotate[0] +
                                     matR(1, 1) * vecWorldPosRotate[1] +
                                     matR(1, 2) * vecWorldPosRotate[2] + vecT[1];
            vecPointTransformed[2] = matR(2, 0) * vecWorldPosRotate[0] +
                                     matR(2, 1) * vecWorldPosRotate[1] +
                                     matR(2, 2) * vecWorldPosRotate[2] + vecT[2];
            //? Mr.Vinh
            float lenVec3D = sqrt(vecPointTransformed[0] * vecPointTransformed[0] +
                                  vecPointTransformed[1] * vecPointTransformed[1] +
                                  vecPointTransformed[2] * vecPointTransformed[2]);
            float lenVec2D = sqrt(vecPointTransformed[0] * vecPointTransformed[0] +
                                  vecPointTransformed[1] * vecPointTransformed[1]);
            float ratio    = 0.0;
            float xd, yd;
            std::vector<int> vecPoint2D(2, 0);
            float theta = 0.0f;
            if (lenVec2D > 1e-6f)
            {
                ratio = lenVec2D / lenVec3D;
                theta = asin(ratio);
                if (vecPointTransformed[2] < 0) theta = M_PI - theta;
                float thetaSq = theta * theta;
                float cdist   = theta * (1.0 + matD[0] * thetaSq + matD[1] * thetaSq * thetaSq +
                                       matD[2] * thetaSq * thetaSq * thetaSq +
                                       matD[3] * thetaSq * thetaSq * thetaSq * thetaSq);
                xd            = vecPointTransformed[0] / lenVec2D * cdist;
                yd            = vecPointTransformed[1] / lenVec2D * cdist;

                vecPoint2D[0] = int((matK(0, 0) * xd + matK(0, 2)) * IMG_W / W);
                vecPoint2D[1] = int((matK(1, 1) * yd + matK(1, 2)) * IMG_H / H);
            }
            else
            {
                vecPoint2D[0] = int(matK(0, 2) * IMG_W / W);
                vecPoint2D[1] = int(matK(1, 2) * IMG_H / H);
            }

            size_t uvIndex = i * LUT_W + j;

            if ((vecPoint2D[1] >= 0.0 && vecPoint2D[1] <= H) &&
                (vecPoint2D[0] >= 0.0 && vecPoint2D[0] <= W))
            {
                if ((camPos == 1 && theta < 90.0 * M_PI / 180) ||
                    (camPos == 2 && theta < 90.0 * M_PI / 180) ||
                    (camPos == 0 && theta < 85.0 * M_PI / 180) ||
                    (camPos == 3 && theta < 85.0 * M_PI / 180))
                {
                    outputFile << int(vecPoint2D[0]) << ", " << int(vecPoint2D[1]) << "\n";
                    uvList.at(uvIndex * 2 + 0) = int(vecPoint2D[0]);
                    uvList.at(uvIndex * 2 + 1) = int(vecPoint2D[1]);
                }
                else
                {
                    uvList.at(uvIndex * 2 + 0) = -2000;
                    uvList.at(uvIndex * 2 + 1) = -2000;
                    outputFile << -2000 << ", " << -2000 << "\n";
                }
            }
            else
            {
                uvList.at(uvIndex * 2 + 0) = -2000;
                uvList.at(uvIndex * 2 + 1) = -2000;
                outputFile << -2000 << ", " << -2000 << "\n";
            }
        }
    }
    outputFile.close();
}
