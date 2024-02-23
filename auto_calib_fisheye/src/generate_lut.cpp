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
    // return value
    // std::vector<std::pair<int, int>> listUVs(TOPVIEW_W * 10 * TOPVIEW_H *
    // 10);

    std::ofstream outputFile(lutOutputFile);

    if (!outputFile.is_open())
    {
        LOG_ERROR("Cannot open file: {}", lutOutputFile);
    }

    for (size_t y = 0; y < TOPVIEW_H * 10; ++y)
    {
        for (size_t x = 0; x < TOPVIEW_W * 10; ++x)
        {
            // float xWorld                         = x / 10.0 - TOPVIEW_W
            // / 2.0; float yWorld                         = y / 10.0 -
            // TOPVIEW_H / 2.0;
            float xWorld = x / 10.0 - TOPVIEW_W / 2.0;
            float yWorld = TOPVIEW_H / 2.0 - y / 10.0;
            // float xWorld                         = TOPVIEW_H / 2.0 - y
            // / 10.0; float yWorld                         = x / 10.0 -
            // TOPVIEW_W / 2.0;
            std::vector<float> vecWorldPosRotate = {xWorld, yWorld, 0};
            float vecPointTransformed[3];
            vecPointTransformed[0] = matR(0, 0) * vecWorldPosRotate[0] +
                                     matR(0, 1) * vecWorldPosRotate[1] +
                                     matR(0, 2) * vecWorldPosRotate[2] +
                                     vecT[0];
            vecPointTransformed[1] = matR(1, 0) * vecWorldPosRotate[0] +
                                     matR(1, 1) * vecWorldPosRotate[1] +
                                     matR(1, 2) * vecWorldPosRotate[2] +
                                     vecT[1];
            vecPointTransformed[2] = matR(2, 0) * vecWorldPosRotate[0] +
                                     matR(2, 1) * vecWorldPosRotate[1] +
                                     matR(2, 2) * vecWorldPosRotate[2] +
                                     vecT[2];
            float lenVec3D =
                sqrt(vecPointTransformed[0] * vecPointTransformed[0] +
                     vecPointTransformed[1] * vecPointTransformed[1] +
                     vecPointTransformed[2] * vecPointTransformed[2]);
            float lenVec2D =
                sqrt(vecPointTransformed[0] * vecPointTransformed[0] +
                     vecPointTransformed[1] * vecPointTransformed[1]);
            float ratio = 0.0;
            float xd, yd;
            std::vector<double> vecPoint2D(2, 0.0);
            float theta = 0.0f;
            if (lenVec3D > 1e-6f)
            {
                ratio = lenVec2D / lenVec3D;
                theta = asin(ratio);
                if (vecPointTransformed[2] < 0) theta = M_PI - theta;
                float thetaSq = theta * theta;
                float cdist =
                    theta *
                    (1.0 + matD[0] * thetaSq + matD[1] * thetaSq * thetaSq +
                     matD[2] * thetaSq * thetaSq * thetaSq +
                     matD[3] * thetaSq * thetaSq * thetaSq * thetaSq);
                xd            = vecPointTransformed[0] / lenVec2D * cdist;
                yd            = vecPointTransformed[1] / lenVec2D * cdist;
                vecPoint2D[0] = matK(0, 0) * xd + matK(0, 2);
                vecPoint2D[1] = matK(1, 1) * yd + matK(1, 2);
            }
            else
            {
                vecPoint2D[0] = matK(0, 2);
                vecPoint2D[1] = matK(1, 2);
            }
            if ((vecPoint2D[1] >= 0.0 && vecPoint2D[1] <= IMG_HEIGHT) &&
                (vecPoint2D[0] >= 0.0 && vecPoint2D[0] <= IMG_WIDTH))
            {
                // listUVs.at(y * TOPVIEW_W * 10 + x) =
                // std::make_pair(int(vecPoint2D[0]), int(vecPoint2D[1]));
                outputFile << int(vecPoint2D[0]) << " " << int(vecPoint2D[1])
                           << "\n";
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
