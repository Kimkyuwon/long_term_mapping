#pragma once

// 모듈 B — 실행 컨텍스트·세션 입출력.
// 01.developerules.mdc 「코드 구조 규칙 (모듈 경계)」 B 항목.
// 전역 그룹 P(경로·스트림)·S(세션 데이터)·A(알고리즘 파라미터) 소유.
// 경로 안전 가드(getDirectory 내부)의 집결 지점이다.

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <gtsam/base/Vector.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "lt_common.hpp"
#include "voxel_evidence.hpp"  // lt_mapping::EvidenceParams

// ---- P: 경로·스트림 ----
extern std::string save_directory, DebugDirectory, ScanDirectory, directory1, directory2, output_directory;
extern std::fstream optimized_stream, edge_stream;

// ---- S: 세션 데이터 ----
// 세션 하나의 자료. dir1_*/dir2_* + First*/Second* "전역 2벌" 패턴을 대신한다
// (02.SW_architecture_rules.mdc 「세션 자료구조」 절). 세션 N개 확장은 Phase 8.
struct SessionData {
    std::string scans_path, poses_path, edges_path, map_path;
    std::vector<double> time;
    std::vector<Pose6> poses;
    std::vector<std::tuple<int, int, gtsam::Vector, Pose6>> edges;
    int size = 0;
};

// sessions[0] = 세션1(구 First*/dir1_*), sessions[1] = 세션2(구 Second*/dir2_*).
extern std::vector<SessionData> sessions;

// ---- A: 알고리즘 파라미터 ----
extern double FOV_u, VOXEL_SIZE;
extern int MAX_DISTANCE;
extern double blind;
extern double dop_thres;
extern double anchor_resolution;
extern double loop_search_radius;
extern lt_mapping::EvidenceParams persistence_params;
extern float session_rigid_residual_thres;
extern float rebuild_pose_shift_thres;

// 잡음 모델 3종(prior/large/odom, `initNoises()` 소비 — map_merge.cpp) 기본값의 단일 출처.
// GTSAM Pose3 6D 접선공간 순서 [회전, 병진]을 따라 rot/trans로 나눈다(01.developerules.mdc
// Lessons Learned #8). 값은 03.task_plan_rules.mdc Phase 1.1 — 변경 금지, 노출만 한다.
extern double noise_prior_rot_variance;
extern double noise_prior_trans_variance;
extern double noise_large_rot_variance;
extern double noise_large_trans_variance;
extern double noise_odom_rot_variance;
extern double noise_odom_trans_variance;

void setParams(std::shared_ptr<rclcpp::Node> nh);
void getDirectory();

pcl::PointCloud<pcl::PointXYZI>::Ptr loadPointCloud(const std::string& filepath);
pcl::PointCloud<pcl::PointXYZI>::Ptr loadRawScanPointCloud(const std::string& scans_dir, int idx);
pcl::PointCloud<pcl::PointXYZI>::Ptr loadProcessedScanPointCloud(const std::string& scans_dir, int idx);

bool loadPoseFile(const std::string& filepath, std::vector<double>& times, std::vector<Pose6>& poses);
bool loadEdgeFile(const std::string& filepath, std::vector<std::tuple<int, int, gtsam::Vector, Pose6>>& edges);
bool loadFiles();

void saveEdge(std::tuple<int, int, gtsam::Vector, Pose6> edge, int idx);
void saveEdges();
