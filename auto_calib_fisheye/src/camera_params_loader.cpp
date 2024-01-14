#include <fstream>
#include <iostream>

#include "camera_params_loader.h"

void getMatrix(Eigen::Matrix4d& matrix, std::vector<double>& rvec,
               std::vector<double>& tvec)
{
    Eigen::Matrix4d rot90;
    rot90 << 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
    double c_y   = cos(rvec[0]);
    double s_y   = sin(rvec[0]);
    double c_r   = cos(rvec[2]);
    double s_r   = sin(rvec[2]);
    double c_p   = cos(rvec[1]);
    double s_p   = sin(rvec[1]);
    matrix(0, 0) = c_p * c_y;
    matrix(0, 1) = c_y * s_p * s_r - s_y * c_r;
    matrix(0, 2) = c_y * s_p * c_r + s_y * s_r;
    matrix(0, 3) = tvec[0] / 1000;
    matrix(1, 0) = s_y * c_p;
    matrix(1, 1) = s_y * s_p * s_r + c_y * c_r;
    matrix(1, 2) = s_y * s_p * c_r - c_y * s_r;
    matrix(1, 3) = tvec[1] / 1000;
    matrix(2, 0) = -s_p;
    matrix(2, 1) = c_p * s_r;
    matrix(2, 2) = c_p * c_r;
    matrix(2, 3) = tvec[2] / 1000;
    matrix(3, 0) = 0;
    matrix(3, 1) = 0;
    matrix(3, 2) = 0;
    matrix(3, 3) = 1;
    matrix       = matrix * rot90;
}

void load_extrinsics(const std::string& extrinsics_dir,
                     Eigen::Matrix4d& left_extrinsics,
                     Eigen::Matrix4d& front_extrinsic,
                     Eigen::Matrix4d& back_extrinsics,
                     Eigen::Matrix4d& right_extrinsics)
{
    std::vector<double> rvec(3);
    std::vector<double> tvec(3);

    // left camera
    std::string filePath = extrinsics_dir + "/extrinsic_cam0.txt";

    std::ifstream inputFile(filePath);
    if (!inputFile.is_open())
    {
        std::cerr << "Error opening the file: " << filePath << std::endl;
        exit(1);
    }

    std::string line;
    // skip first two line
    std::getline(inputFile, line);
    std::getline(inputFile, line);

    // get rvec
    std::getline(inputFile, line);
    rvec[0] = std::stod(line);
    std::getline(inputFile, line);
    rvec[1] = std::stod(line);
    std::getline(inputFile, line);
    rvec[2] = std::stod(line);

    // get tvec
    std::getline(inputFile, line);
    std::getline(inputFile, line);
    tvec[0] = std::stod(line);
    std::getline(inputFile, line);
    tvec[1] = std::stod(line);
    std::getline(inputFile, line);
    tvec[2] = std::stod(line);
    inputFile.close();
    inputFile.clear();
    getMatrix(left_extrinsics, rvec, tvec);

    // front camera
    filePath = extrinsics_dir + "/extrinsic_cam1.txt";

    inputFile.open(filePath);
    if (!inputFile.is_open())
    {
        std::cerr << "Error opening the file: " << filePath << std::endl;
        exit(1);
    }

    // skip first two line
    std::getline(inputFile, line);
    std::getline(inputFile, line);

    // get rvec
    std::getline(inputFile, line);
    rvec[0] = std::stod(line);
    std::getline(inputFile, line);
    rvec[1] = std::stod(line);
    std::getline(inputFile, line);
    rvec[2] = std::stod(line);

    // get tvec
    std::getline(inputFile, line);
    std::getline(inputFile, line);
    tvec[0] = std::stod(line);
    std::getline(inputFile, line);
    tvec[1] = std::stod(line);
    std::getline(inputFile, line);
    tvec[2] = std::stod(line);
    inputFile.close();
    inputFile.clear();
    getMatrix(front_extrinsic, rvec, tvec);

    // back camera
    filePath = extrinsics_dir + "/extrinsic_cam2.txt";

    inputFile.open(filePath);
    if (!inputFile.is_open())
    {
        std::cerr << "Error opening the file: " << filePath << std::endl;
        exit(1);
    }

    // skip first two line
    std::getline(inputFile, line);
    std::getline(inputFile, line);

    // get rvec
    std::getline(inputFile, line);
    rvec[0] = std::stod(line);
    std::getline(inputFile, line);
    rvec[1] = std::stod(line);
    std::getline(inputFile, line);
    rvec[2] = std::stod(line);

    // get tvec
    std::getline(inputFile, line);
    std::getline(inputFile, line);
    tvec[0] = std::stod(line);
    std::getline(inputFile, line);
    tvec[1] = std::stod(line);
    std::getline(inputFile, line);
    tvec[2] = std::stod(line);
    inputFile.close();
    inputFile.clear();
    getMatrix(back_extrinsics, rvec, tvec);

    // right camera
    filePath = extrinsics_dir + "/extrinsic_cam3.txt";

    inputFile.open(filePath);
    if (!inputFile.is_open())
    {
        std::cerr << "Error opening the file: " << filePath << std::endl;
        exit(1);
    }

    // skip first two line
    std::getline(inputFile, line);
    std::getline(inputFile, line);

    // get rvec
    std::getline(inputFile, line);
    rvec[0] = std::stod(line);
    std::getline(inputFile, line);
    rvec[1] = std::stod(line);
    std::getline(inputFile, line);
    rvec[2] = std::stod(line);

    // get tvec
    std::getline(inputFile, line);
    std::getline(inputFile, line);
    tvec[0] = std::stod(line);
    std::getline(inputFile, line);
    tvec[1] = std::stod(line);
    std::getline(inputFile, line);
    tvec[2] = std::stod(line);
    inputFile.close();
    inputFile.clear();
    getMatrix(right_extrinsics, rvec, tvec);
}
