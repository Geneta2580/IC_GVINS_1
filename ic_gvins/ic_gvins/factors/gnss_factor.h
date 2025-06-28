/*
 * IC-GVINS: A Robust, Real-time, INS-Centric GNSS-Visual-Inertial Navigation System
 *
 * Copyright (C) 2022 i2Nav Group, Wuhan University
 *
 *     Author : Hailiang Tang
 *    Contact : thl@whu.edu.cn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef GNSS_FACTOR_H
#define GNSS_FACTOR_H

#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <cmath>

#include "common/types.h"

class GnssFactor : public ceres::SizedCostFunction<3, 7> {

public:
    explicit GnssFactor(GNSS gnss, Vector3d lever)
        : gnss_(std::move(gnss))
        , lever_(std::move(lever)) {
    }

    void updateGnssState(const GNSS &gnss) {
        gnss_ = gnss;
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        Vector3d p{parameters[0][0], parameters[0][1], parameters[0][2]};
        Quaterniond q{parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]};

        Eigen::Map<Eigen::Matrix<double, 3, 1>> error(residuals);

        error = p + q.toRotationMatrix() * lever_ - gnss_.blh; // 这里的gnss.blh已经是局部坐标系了，因为前面addNewGNSS里做过转换

        Matrix3d sqrt_info_ = Matrix3d::Zero();
        sqrt_info_(0, 0)    = 1.0 / gnss_.std[0]; // 信息矩阵赋值，std分别对应GNSS量测的0\4\8位
        sqrt_info_(1, 1)    = 1.0 / gnss_.std[1];
        sqrt_info_(2, 2)    = 1.0 / gnss_.std[2];

        error = sqrt_info_ * error;

        // Vector3d error_1 = sqrt_info_ * error; // error经过信息矩阵加权，这里的error就是残差

        // error = compute_weighted_error(error_1); // 根据GMM模型重新进行权重分配

        // error = Eigen::Vector3d::Zero(); // 屏蔽GNSS残差因子对优化的影响

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jacobian_pose(jacobians[0]);
                jacobian_pose.setZero();

                jacobian_pose.block<3, 3>(0, 0) = Matrix3d::Identity();
                jacobian_pose.block<3, 3>(0, 3) = -q.toRotationMatrix() * Rotation::skewSymmetric(lever_);

                jacobian_pose = sqrt_info_ * jacobian_pose;
            }
        }

        return true;
    }

    // double gaussian_pdf(double x, double mu, double sigma) const {    
    //     const double norm = 1.0 / sqrt(2 * M_PI * sigma);
    //     const double exponent = -0.5 * pow((x - mu), 2) / sigma;
    //     return norm * exp(exponent);
    // }

    // // 计算后验概率加权误差
    // Vector3d compute_weighted_error(const Vector3d& error) const {
    //     // 计算各维度后验概率
    //     double post_probs[3]; // 存储三个维度在两个成分的概率
        
    //     // E向计算
    //     double e_prob0 = 0.3448 * gaussian_pdf(error[0], 0.2219, 3.36);
    //     double e_prob1 = 0.6552 * gaussian_pdf(error[0], -16.8896, 1.0198e4);
    //     post_probs[0] = e_prob0 / (e_prob0 + e_prob1); // 成分1的后验概率
        
    //     // N向计算(注意第二个成分均值偏移)
    //     double n_prob0 = 0.5072 * gaussian_pdf(error[1], -1.8278, 40.5442);
    //     double n_prob1 = 0.4928 * gaussian_pdf(error[1], 19.1208, 1.6267e4); 
    //     post_probs[1] = n_prob0 / (n_prob0 + n_prob1);
        
    //     // U向计算
    //     double u_prob0 = 0.7415 * gaussian_pdf(error[2], 0.0331, 19.1084);
    //     double u_prob1 = 0.2585 * gaussian_pdf(error[2], -49.5925, 1.0759e5);
    //     post_probs[2] = u_prob0 / (u_prob0 + u_prob1);

    //     // 计算联合后验权重(假设各维度独立)
    //     Vector3d weight(post_probs[0], post_probs[1], post_probs[2]);
        
    //     // 加权误差计算(信息矩阵已通过sqrt_info_预乘)
    //     Vector3d weighted_error;
    //     weighted_error = error.cwiseProduct(weight) * 0.4; // 调整系数
        
    //     return weighted_error;
    // }

private:
    GNSS gnss_;
    Vector3d lever_;
};

#endif // GNSS_FACTOR_H
