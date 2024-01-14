#/*
 * Copyright (c) 2020 - 2021, VinAI. All rights reserved. All information
 * information contained herein is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */

#include <Eigen/Core>

#ifndef SVM_CAMERA_PARAMS_LOADER
#define SVM_CAMERA_PARAMS_LOADER

Eigen::MatrixXd getMatrix(double yaw, double pitch, double roll);

void load_extrinsics(const std::string& extrinsics_dir,
                     Eigen::Matrix4d& left_extrinsics,
                     Eigen::Matrix4d& front_extrinsic,
                     Eigen::Matrix4d& back_extrinsics,
                     Eigen::Matrix4d& right_extrinsics);

#endif  // SVM_CAMERA_PARAMS_LOADER
