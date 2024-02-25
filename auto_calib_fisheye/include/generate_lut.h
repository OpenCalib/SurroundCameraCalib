#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "defines.h"

namespace lut {

void genLUT(const Eigen::Matrix3d& matR, const Eigen::Vector3d& vecT,
            const Eigen::Matrix3d& matK, const std::vector<double>& matD,
            const std::string& lutOutputFile);

void genLUT(CamID camPos, const Eigen::Matrix3d& matR,
            const Eigen::Vector3d& vecT, const Eigen::Matrix3d& matK,
            const std::vector<double>& matD, std::vector<short>& uvList,
            const std::string& lutOutputFile, bool isSeg);

}  // namespace lut
