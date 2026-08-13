#pragma once

// 모듈 C — 지도 병합·최적화·병합지도 산출.
// 01.developerules.mdc 「코드 구조 규칙 (모듈 경계)」 C 항목.
// 전역 그룹 G(포즈그래프 상태)·V(시각화, PubMerge_map은 모듈 D 소유) 소유.
// common_lib.h는 이 모듈의 .cpp에서만 include한다(ODR 회피, 01.developerules.mdc 참고).

#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <nano_gicp/nano_gicp.hpp>
#include <nano_gicp/point_type_nano_gicp.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include "lt_common.hpp"

// ---- G: 포즈그래프 상태 ----
extern nano_gicp::NanoGICP<PointType2, PointType2> gicp;
extern pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree;

extern gtsam::NonlinearFactorGraph gtSAMgraph;
extern bool gtSAMgraphMade;
extern gtsam::Values initialEstimate;
extern gtsam::ISAM2* isam;
extern gtsam::Values isamCurrentEstimate;
extern gtsam::Vector odomNoiseVector6;
extern gtsam::Vector robustNoiseVector6;
extern gtsam::noiseModel::Diagonal::shared_ptr priorNoise;
extern gtsam::noiseModel::Diagonal::shared_ptr largeNoise;
extern gtsam::noiseModel::Diagonal::shared_ptr odomNoise;
extern gtsam::noiseModel::Base::shared_ptr robustLoopNoise;
extern gtsam::Pose3 A2_anchor;
extern std::vector<std::pair<int, int>> loop_pairs;
extern std::set<std::pair<int, int>> first_session_edge_pairs;
extern std::set<std::pair<int, int>> second_session_edge_pairs;

// [Phase2.0] 항목7(루프 엣지 GICP 정합 품질) 실측 전용 진단 버퍼. 그래프 구성에는 영향 없음
struct LoopGicpDiagnostic {
    bool converged;
    double fitness_score;
    double matching_dop;
    double dop_ratio;
    bool accepted;
};
extern std::vector<LoopGicpDiagnostic> loop_gicp_diagnostics;

struct SessionRigidResidualStats {
    int session_id = 0;
    size_t sample_count = 0;
    double rms = 0.0;
    double max_residual = 0.0;
    double rigid_translation_norm = 0.0;
    double rigid_rotation_deg = 0.0;
};

// ---- V: 시각화 (PubMerge_map은 모듈 D 소유) ----
extern visualization_msgs::msg::Marker loopLine;
extern nav_msgs::msg::Path FirstMap_path, SecondMap_path, Merge_path;

extern pcl::PointCloud<pcl::PointXYZI> FirstMap_nodes, SecondMap_nodes;
extern rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr First_kf_node_pub, Second_kf_node_pub;
extern rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr LoopLineMarker_pub;
extern rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr PubFirstMap_path, PubSecondMap_path, PubMerge_path;
extern rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr completion_pub;

// 퍼블리셔 초기화/해제 및 ISAM2 정리 — main()의 반복 코드를 모듈 C 안으로 모음.
void initMapMergePublishers(const std::shared_ptr<rclcpp::Node>& nh, const rclcpp::QoS& qos_viz);
void resetMapMergePublishers();
void cleanupIsam();

void initNoises();

// 세션 로컬 노드의 전역 포즈 = 그 세션의 앵커 ∘ 로컬 노드 값.
gtsam::Pose3 getGlobalPose(int session_idx, int node_idx);

SessionRigidResidualStats evaluateSessionRigidity(
    int session_id,
    const std::vector<Pose6>& original_poses,
    const std::vector<Pose6>& optimized_poses);

void logSessionRigidityStats(const SessionRigidResidualStats& stats, float threshold, float shift_threshold);

// [Phase2.0] 항목7 실측 전용 — 루프 후보 전체(accept+reject)의 GICP 수렴률/적합도 요약. 그래프 구성 로직과 무관
void logLoopGicpDiagnostics();

double computeDOP(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, Eigen::Vector3d pos);

std::optional<gtsam::Pose3> doGICPVirtualRelative(int _loop_kf_idx, int _curr_kf_idx, Eigen::Matrix4f delta_TF);

void updatePoses(void);

void runISAM2opt(void);

std::vector<Pose6> collectOptimizedSessionPoses(int session_id, int session_size);

void generateOptimizedMap();

void getEdges();

void getLoopEdges();

void getPoses();
