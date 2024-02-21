#include <Eigen/Core>
#include <Eigen/Dense>

#define IMG_HEIGHT 800
#define IMG_WIDTH  1280

namespace lut {

void genLUT(const Eigen::Matrix3d& matR,
                                        const Eigen::Vector3d& vecT,
                                        const Eigen::Matrix3d& matK,
                                        const std::vector<double>& matD,
                                        const std::string& lutOutputFile);

}
