#pragma once

// 모듈 A — 공통 타입·기하 변환. 헤더 온리, 전역 미접촉 순수 계산만 담는다.
// 01.developerules.mdc 「코드 구조 규칙 (모듈 경계)」 A 항목.

#include <cstdint>
#include <cmath>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>

struct Pose6 {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

inline int64_t packKey(int ix, int iy, int iz)
{
    return ((int64_t)(ix & 0x1FFFFF)) |
           ((int64_t)(iy & 0x1FFFFF) << 21) |
           ((int64_t)(iz & 0x1FFFFF) << 42);
}

inline int64_t pointToKey(const pcl::PointXYZI& pt, double resolution)
{
    int ix = static_cast<int>(std::floor(pt.x / resolution));
    int iy = static_cast<int>(std::floor(pt.y / resolution));
    int iz = static_cast<int>(std::floor(pt.z / resolution));
    return packKey(ix, iy, iz);
}

inline std::vector<Eigen::Vector3f> convertCloudToVec(const pcl::PointCloud<pcl::PointXYZI>& cloud) {
    std::vector<Eigen::Vector3f> vec;
    vec.reserve(cloud.size());
    for (const auto& pt : cloud.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      vec.emplace_back(pt.x, pt.y, pt.z);
    }
    return vec;
}

// Quaternion을 Euler angles로 변환하는 함수
inline Pose6 poseToPose6(double x, double y, double z, double qx, double qy, double qz, double qw)
{
    Pose6 pose;
    pose.x = x;
    pose.y = y;
    pose.z = z;
    
    // Quaternion to Euler angles conversion
    tf2::Quaternion q(qx, qy, qz, qw);
    tf2::Matrix3x3 m(q);
    m.getRPY(pose.roll, pose.pitch, pose.yaw);
    
    return pose;
}

inline Eigen::Matrix4f get_TF_Matrix(const Pose6 Pose)
{
    Eigen::Matrix3f rotation;
    rotation = Eigen::AngleAxisf(Pose.yaw, Eigen::Vector3f::UnitZ())
             * Eigen::AngleAxisf(Pose.pitch, Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(Pose.roll, Eigen::Vector3f::UnitX());
    Eigen::Matrix4f TF(Eigen::Matrix4f::Identity());
    TF.block(0,0,3,3) = rotation;
    TF(0,3) = Pose.x;
    TF(1,3) = Pose.y;
    TF(2,3) = Pose.z;

    return TF;
}

inline gtsam::Pose3 Pose6toGTSAMPose3(const Pose6& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6toGTSAMPose3

inline int getGlobalNodeIdx(int session_idx, int node_idx)
{
    return (session_idx * 1000000) + node_idx;
}

// getGlobalNodeIdx(session*1e6+idx)와 겹치지 않는 오프셋. session_idx는 1~수십 범위이므로
// 9억을 더해도 getGlobalNodeIdx가 만들 수 있는 최대값(대략 수십*1e6)과 충돌하지 않는다.
constexpr int kAnchorNodeIdxBase = 900000000;

// 세션 앵커(세션 로컬 → 전역) 변환을 담는 그래프 변수의 키.
// LT-mapper(ltslam) Form A: 앵커를 노드 값에 미리 곱해 넣지 않고 별도 변수로 명시한다
// (utility.cpp의 genAnchorNodeIdx와 동일한 설계).
inline int getAnchorNodeIdx(int session_idx)
{
    return kAnchorNodeIdxBase + session_idx;
}

inline Eigen::Matrix4f createTransformMatrix(const Pose6& pose)
{
    Eigen::Matrix3f rotation = (Eigen::AngleAxisf(pose.yaw, Eigen::Vector3f::UnitZ())
                              * Eigen::AngleAxisf(pose.pitch, Eigen::Vector3f::UnitY())
                              * Eigen::AngleAxisf(pose.roll, Eigen::Vector3f::UnitX())).toRotationMatrix();
    
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block(0,0,3,3) = rotation;
    transform(0,3) = pose.x;
    transform(1,3) = pose.y;
    transform(2,3) = pose.z;
    
    return transform;
}
