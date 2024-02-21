#include <Eigen/Core>
#include <Eigen/Dense>

#define IMG_HEIGHT 800
#define IMG_WIDTH  1280

namespace lut {

std::vector<std::vector<int>> genLUT(
    const Eigen::Matrix3d& matR, const Eigen::Vector3d& vecT,
    const Eigen::Matrix3d& matK, const Eigen::Vector4d& matD);

}
