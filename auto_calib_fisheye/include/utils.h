/*
 * Copyright (c) 2020 - 2021, VinAI. All rights reserved. All information
 * information contained herein is proprietary and confidential to VinAI.
 * Any use, reproduction, or disclosure without the written permission
 * of VinAI is prohibited.
 */

#ifndef SVM_AUTORC_UTILS_H
#define SVM_AUTORC_UTILS_H

#include <Eigen/Dense>
#include "defines.h"
#include "optimizer.h"
#include "transform_util.h"

namespace util {

inline std::pair<double, double> calculateError(const Eigen::Matrix4d& ext1,
                                                const Eigen::Matrix4d& ext2)
{
    // translation error
    Eigen::Vector3d translation1 = ext1.block<3, 1>(0, 3);
    Eigen::Vector3d translation2 = ext2.block<3, 1>(0, 3);
    double translationError      = (translation1 - translation2).norm();

    // rotation error
    Eigen::Matrix3d rotation1 = ext1.block<3, 3>(0, 0);
    Eigen::Matrix3d rotation2 = ext2.block<3, 3>(0, 0);
    double trace              = (rotation1.transpose() * rotation2).trace();
    double rotationError      = std::acos((trace - 1.0) / 2.0) * 180 / M_PI;
    return std::make_pair(translationError, rotationError);
}

inline void addDisturbance(CamID fixed, Eigen::Matrix4d& T_FG,
                           Eigen::Matrix4d& T_LG, Eigen::Matrix4d& T_BG,
                           Eigen::Matrix4d& T_RG)
{
    if (fixed == CamID::B)
    {
        Eigen::Matrix4d front_disturbance;
        Eigen::Matrix3d front_disturbance_rot_mat;
        Vec3f front_disturbance_rot_euler;  // R(euler)
        Mat_<double> front_disturbance_t =
            (Mat_<double>(3, 1) << 0.007, 0.008, -0.0093);
        front_disturbance_rot_euler << 0.89, 2.69, 1.05;
        front_disturbance_rot_mat = TransformUtil::eulerAnglesToRotationMatrix(
            front_disturbance_rot_euler);
        front_disturbance = TransformUtil::R_T2RT(
            TransformUtil::eigen2mat(front_disturbance_rot_mat),
            front_disturbance_t);
        T_FG *= front_disturbance;
    }

    Eigen::Matrix4d left_disturbance;
    Eigen::Matrix3d left_disturbance_rot_mat;
    Vec3f left_disturbance_rot_euler;  // R(euler)
    // Mat_<double> left_disturbance_t=(Mat_<double>(3, 1)<<0,0,0);
    Mat_<double> left_disturbance_t =
        (Mat_<double>(3, 1) << 0.0095, 0.0025, -0.0086);
    left_disturbance_rot_euler << 1.95, -1.25, 1.86;
    left_disturbance_rot_mat =
        TransformUtil::eulerAnglesToRotationMatrix(left_disturbance_rot_euler);
    left_disturbance = TransformUtil::R_T2RT(
        TransformUtil::eigen2mat(left_disturbance_rot_mat), left_disturbance_t);
    T_LG *= left_disturbance;

    Eigen::Matrix4d right_disturbance;
    Eigen::Matrix3d right_disturbance_rot_mat;
    Vec3f right_disturbance_rot_euler;
    // Mat_<double> right_disturbance_t=(Mat_<double>(3, 1)<<0,0,0);
    Mat_<double> right_disturbance_t =
        (Mat_<double>(3, 1) << 0.0065, -0.0075, 0.0095);
    right_disturbance_rot_euler << 1.95, 0.95, -1.8;
    right_disturbance_rot_mat =
        TransformUtil::eulerAnglesToRotationMatrix(right_disturbance_rot_euler);
    right_disturbance = TransformUtil::R_T2RT(
        TransformUtil::eigen2mat(right_disturbance_rot_mat),
        right_disturbance_t);
    T_RG *= right_disturbance;

    if (fixed == CamID::F)
    {
        Eigen::Matrix4d behind_disturbance;
        Eigen::Matrix3d behind_disturbance_rot_mat;
        Vec3f behind_disturbance_rot_euler;
        // Mat_<double> behind_disturbance_t=(Mat_<double>(3, 1)<<0,0,0);
        Mat_<double> behind_disturbance_t =
            (Mat_<double>(3, 1) << -0.002, -0.0076, 0.0096);
        behind_disturbance_rot_euler << -1.75, 1.95, -1.8;
        behind_disturbance_rot_mat = TransformUtil::eulerAnglesToRotationMatrix(
            behind_disturbance_rot_euler);
        behind_disturbance = TransformUtil::R_T2RT(
            TransformUtil::eigen2mat(behind_disturbance_rot_mat),
            behind_disturbance_t);
        T_BG *= behind_disturbance;
    }
}

}  // namespace util

#endif  // SVM_AUTORC_UTILS_H
