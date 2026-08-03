#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <queue>
#include <fstream>
#include <csignal>
#include <optional>
#include <chrono>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <array>
#include <algorithm>
#include <limits>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/console/print.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d_omp.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/utils.h>
#include <geometry_msgs/msg/vector3.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <common_lib.h>
#include "solid/solid_module.h"
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <nano_gicp/point_type_nano_gicp.hpp>
#include <nano_gicp/nano_gicp.hpp>
#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/GncSolver.hpp>
#include <kiss_matcher/KISSMatcher.hpp>

#include <ufo/map/ufomap.hpp>
#include "voxel_evidence.hpp"

#define DOP_VOXEL_SIZE      (2.5)
#define MEAN_RANGE          (10.0)

using namespace std;
using namespace gtsam;

struct Pose6 {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
};

// 두 디렉토리 명을 파라미터 또는 터미널에서 입력받기
string save_directory, DebugDirectory, ScanDirectory, directory1, directory2, output_directory;
string dir1_scans_path, dir1_poses_path, dir1_edges_path, dir1_map_path, dir2_scans_path, dir2_poses_path, dir2_edges_path, dir2_map_path;

fstream optimized_stream, edge_stream;

// 시간과 포즈 데이터를 저장할 벡터들
std::vector<double> FirstMapTime, SecondMapTime;
std::vector<Pose6> FirstMapPoses, SecondMapPoses, MergeMapPoses;
std::vector<tuple<int, int, gtsam::Vector, Pose6>> FirstMapEdges, SecondMapEdges;
int FirstMapSize, SecondMapSize, MergeMapSize;

// boost::shared_ptr<PatchWorkpp<pcl::PointXYZI>> PatchworkppGroundSeg;

// solid params
SOLiDModule solidModule;
double R_SOLiD_THRES;
double FOV_u, FOV_d, VOXEL_SIZE;
int NUM_ANGLE, NUM_RANGE, NUM_HEIGHT;
int MIN_DISTANCE, MAX_DISTANCE, NUM_EXCLUDE_RECENT, NUM_CANDIDATES_FROM_TREE;
vector<tuple<int, int, double>> solidLoopBuf; 

double blind;

// edge measurement params
nano_gicp::NanoGICP<PointType2, PointType2> gicp;
std::vector<int> pointSearchInd;
std::vector<float> pointSearchSqDis;
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtree (new pcl::KdTreeFLANN<pcl::PointXYZI>());
std::vector<int> indiceLet;
double dop_thres = 0;
lt_mapping::EvidenceParams persistence_params;

//for pose graph
gtsam::NonlinearFactorGraph gtSAMgraph;
bool gtSAMgraphMade = false;
bool isLoopClosed = false;
gtsam::Values initialEstimate;
gtsam::ISAM2 *isam;
gtsam::Values isamCurrentEstimate;
gtsam::Vector odomNoiseVector6(6);
gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
noiseModel::Diagonal::shared_ptr priorNoise;
noiseModel::Diagonal::shared_ptr largeNoise;
noiseModel::Diagonal::shared_ptr odomNoise;
noiseModel::Base::shared_ptr robustLoopNoise;
int recentIdxUpdated = 0;
gtsam::Pose3 A2_anchor;
double anchor_resolution;
vector<pair<int, int>> loop_pairs;

visualization_msgs::msg::Marker loopLine;
nav_msgs::msg::Path FirstMap_path, SecondMap_path, Merge_path;

pcl::PointCloud<pcl::PointXYZI> FirstMap_nodes, SecondMap_nodes;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr First_kf_node_pub, Second_kf_node_pub, Merge_kf_node_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr LoopLineMarker_pub;
rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr PubFirstMap_path, PubSecondMap_path, PubMerge_path;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr PubMerge_map;
rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr completion_pub;

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

void setParams (std::shared_ptr<rclcpp::Node> nh)
{

    // PatchworkppGroundSeg.reset(new PatchWorkpp<pcl::PointXYZI>());

    nh->declare_parameter("blind", 0.01);
    nh->declare_parameter("r_solid_thres", 0.99);
    nh->declare_parameter("fov_u", 2.0);
    nh->declare_parameter("fov_d", -24.8);
    nh->declare_parameter("num_angle", 60);
    nh->declare_parameter("num_range", 40);
    nh->declare_parameter("num_height", 32);
    nh->declare_parameter("min_distance", 3);
    nh->declare_parameter("max_distance", 80);
    nh->declare_parameter("anchor_resolution", 1.0);
    nh->declare_parameter("voxel_size", 0.4);
    nh->declare_parameter("num_exclude_recent", 30);
    nh->declare_parameter("num_candidates_from_tree", 3);
    nh->declare_parameter("dop_thres", 0.5);
    nh->declare_parameter("directory1", std::string(""));
    nh->declare_parameter("directory2", std::string(""));
    nh->declare_parameter("output_directory", std::string(""));

    nh->declare_parameter("persistence.res", 0.2);
    nh->declare_parameter("persistence.r0", 15.0);
    nh->declare_parameter("persistence.r_s", 20.0);
    nh->declare_parameter("persistence.d_surf", 0.5);
    nh->declare_parameter("persistence.aniso_typ", 0.3);
    nh->declare_parameter("persistence.eps_reg", 1e-3);
    nh->declare_parameter("persistence.vdop_typ", 1.5);
    nh->declare_parameter("persistence.sigma_vdop", 1.0);
    nh->declare_parameter("persistence.n_req", 3.0);
    nh->declare_parameter("persistence.w_min", 0.05);
    nh->declare_parameter("persistence.n_sat", 5.0);
    nh->declare_parameter("persistence.l_hit", 0.85);
    nh->declare_parameter("persistence.l_void", 0.4);
    nh->declare_parameter("persistence.l_max", 5.0);
    nh->declare_parameter("persistence.tau_del", 0.3);
    nh->declare_parameter("persistence.tau_add", 0.7);
    nh->declare_parameter("persistence.cross_dilate", 1);
    nh->declare_parameter("persistence.dilate_weight", 0.5);
    nh->declare_parameter("persistence.cluster_eps", 0.5);
    nh->declare_parameter("persistence.min_cluster_size", 11);
    
    nh->get_parameter_or<double>("blind", blind, 0.01);
    nh->get_parameter_or<double>("r_solid_thres", R_SOLiD_THRES, 0.99);
    nh->get_parameter_or<double>("fov_u", FOV_u, 2.0);
    nh->get_parameter_or<double>("fov_d", FOV_d, -24.8);
    nh->get_parameter_or<int>("num_angle", NUM_ANGLE, 60);
    nh->get_parameter_or<int>("num_range", NUM_RANGE, 40);
    nh->get_parameter_or<int>("num_height", NUM_HEIGHT, 32);
    nh->get_parameter_or<int>("min_distance", MIN_DISTANCE, 3);
    nh->get_parameter_or<int>("max_distance", MAX_DISTANCE, 80);
    nh->get_parameter_or<double>("anchor_resolution", anchor_resolution, 1.0);
    nh->get_parameter_or<double>("voxel_size", VOXEL_SIZE, 0.4);
    nh->get_parameter_or<int>("num_exclude_recent", NUM_EXCLUDE_RECENT, 30);
    nh->get_parameter_or<int>("num_candidates_from_tree", NUM_CANDIDATES_FROM_TREE, 3);
    nh->get_parameter_or<double>("dop_thres", dop_thres, 0.5);

    nh->get_parameter_or<float>("persistence.res", persistence_params.res, 0.2f);
    nh->get_parameter_or<float>("persistence.r0", persistence_params.r0, 15.0f);
    nh->get_parameter_or<float>("persistence.r_s", persistence_params.r_s, 20.0f);
    nh->get_parameter_or<float>("persistence.d_surf", persistence_params.d_surf, 0.5f);
    nh->get_parameter_or<float>("persistence.aniso_typ", persistence_params.aniso_typ, 0.3f);
    nh->get_parameter_or<float>("persistence.eps_reg", persistence_params.eps_reg, 1e-3f);
    nh->get_parameter_or<float>("persistence.vdop_typ", persistence_params.vdop_typ, 1.5f);
    nh->get_parameter_or<float>("persistence.sigma_vdop", persistence_params.sigma_vdop, 1.0f);
    nh->get_parameter_or<float>("persistence.n_req", persistence_params.n_req, 3.0f);
    nh->get_parameter_or<float>("persistence.w_min", persistence_params.w_min, 0.05f);
    nh->get_parameter_or<float>("persistence.n_sat", persistence_params.n_sat, 5.0f);
    nh->get_parameter_or<float>("persistence.l_hit", persistence_params.l_hit, 0.85f);
    nh->get_parameter_or<float>("persistence.l_void", persistence_params.l_void, 0.4f);
    nh->get_parameter_or<float>("persistence.l_max", persistence_params.l_max, 5.0f);
    nh->get_parameter_or<float>("persistence.tau_del", persistence_params.tau_del, 0.3f);
    nh->get_parameter_or<float>("persistence.tau_add", persistence_params.tau_add, 0.7f);
    nh->get_parameter_or<int>("persistence.cross_dilate", persistence_params.cross_dilate, 1);
    nh->get_parameter_or<float>("persistence.dilate_weight", persistence_params.dilate_weight, 0.5f);
    nh->get_parameter_or<float>("persistence.cluster_eps", persistence_params.cluster_eps, 0.5f);
    nh->get_parameter_or<int>("persistence.min_cluster_size", persistence_params.min_cluster_size, 11);

    // 먼저 ROS2 파라미터에서 값 확인
    nh->get_parameter_or<std::string>("directory1", directory1, std::string(""));
    nh->get_parameter_or<std::string>("directory2", directory2, std::string(""));
    nh->get_parameter_or<std::string>("output_directory", output_directory, std::string(""));

    std::cout << "=== Parameters loaded ===" << std::endl;
    std::cout << "directory1: " << directory1 << std::endl;
    std::cout << "directory2: " << directory2 << std::endl;
    std::cout << "output_directory: " << output_directory << std::endl;

    
    solidModule.setParams(FOV_u, FOV_d, NUM_ANGLE, NUM_RANGE, NUM_HEIGHT, MIN_DISTANCE, MAX_DISTANCE, VOXEL_SIZE, NUM_EXCLUDE_RECENT, NUM_CANDIDATES_FROM_TREE, R_SOLiD_THRES);

    gicp.setMaxCorrespondenceDistance(2.0);
    gicp.setNumThreads(2);
    gicp.setCorrespondenceRandomness(15);
    gicp.setMaximumIterations(3);
    gicp.setTransformationEpsilon(0.01);
    gicp.setEuclideanFitnessEpsilon(0.01);
    gicp.setRANSACOutlierRejectionThreshold(1.0);

    loopLine.type = visualization_msgs::msg::Marker::LINE_LIST;
    loopLine.action = visualization_msgs::msg::Marker::ADD;
    loopLine.color.b = 1.0; loopLine.color.a = 0.5;
    loopLine.scale.x = 0.1;
    loopLine.header.frame_id = "map";

    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new ISAM2(parameters);

}

void initNoises( void )
{
    gtsam::Vector priorNoiseVector6(6);
    priorNoiseVector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    priorNoise = noiseModel::Diagonal::Variances(priorNoiseVector6);

    gtsam::Vector largeNoiseVector6(6);
    largeNoiseVector6 << M_PI*M_PI, M_PI*M_PI, M_PI*M_PI, 1e8, 1e8, 1e8;
    largeNoise = noiseModel::Diagonal::Variances(largeNoiseVector6);

    odomNoiseVector6 << 1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2;
    odomNoise = noiseModel::Diagonal::Variances(odomNoiseVector6);

    double loopNoiseScore = 0.5; // constant is ok...
    robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    robustLoopNoise = gtsam::noiseModel::Robust::Create(
                    gtsam::noiseModel::mEstimator::Cauchy::Create(1.0), // optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
                    gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6) );
} // initNoises

std::vector<Eigen::Vector3f> convertCloudToVec(const pcl::PointCloud<pcl::PointXYZI>& cloud) {
    std::vector<Eigen::Vector3f> vec;
    vec.reserve(cloud.size());
    for (const auto& pt : cloud.points) {
      if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
      vec.emplace_back(pt.x, pt.y, pt.z);
    }
    return vec;
}

// Quaternion을 Euler angles로 변환하는 함수
Pose6 poseToPose6(double x, double y, double z, double qx, double qy, double qz, double qw)
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

Eigen::Matrix4f get_TF_Matrix(const Pose6 Pose)
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

gtsam::Pose3 Pose6toGTSAMPose3(const Pose6& p)
{
    return gtsam::Pose3( gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z) );
} // Pose6toGTSAMPose3

int getGlobalNodeIdx(int session_idx, int node_idx)
{
    return (session_idx * 1000000) + node_idx;
}

Eigen::Matrix4f createTransformMatrix(const Pose6& pose)
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

pcl::PointCloud<pcl::PointXYZI>::Ptr loadPointCloud(const std::string& filepath)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    if (pcl::io::loadPCDFile(filepath, *cloud) == -1) {
        RCLCPP_ERROR(rclcpp::get_logger("posegraphoptimization"), "Failed to load point cloud: %s", filepath.c_str());
    }
    return cloud;
}

double computeDOP(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, Eigen::Vector3d pos)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr dop_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterDOP;
    downSizeFilterDOP.setLeafSize(DOP_VOXEL_SIZE, DOP_VOXEL_SIZE, DOP_VOXEL_SIZE);
    downSizeFilterDOP.setInputCloud(cloud);
    downSizeFilterDOP.filter(*dop_cloud);  
    pcl::removeNaNFromPointCloud(*dop_cloud, *dop_cloud, indiceLet);
    indiceLet.clear();

    std::vector<Eigen::Vector3d> range_info;
    for (size_t k = 0; k < dop_cloud->points.size(); k++)
    {
        double r = sqrt(pow((dop_cloud->points[k].x-pos(0)), 2) + pow((dop_cloud->points[k].y-pos(1)), 2) + pow((dop_cloud->points[k].z-pos(2)), 2));
        if (r < blind)    continue;
        Eigen::Vector3d r_info;
        r_info(0) = (dop_cloud->points[k].x-pos(0)) / r;
        r_info(1) = (dop_cloud->points[k].y-pos(1)) / r;
        r_info(2) = (dop_cloud->points[k].z-pos(2)) / r;
        range_info.push_back(r_info);    
    }
    Eigen::MatrixXd AA(range_info.size(), 3);
    for (size_t p = 0; p < range_info.size(); p++)
    {
        AA(p, 0) = range_info[p](0);
        AA(p, 1) = range_info[p](1);
        AA(p, 2) = range_info[p](2);
    }
    Eigen::Matrix3d A_sq;
    Eigen::Matrix3d Q;
    A_sq = AA.transpose() * AA;
    Q = A_sq.inverse();

    double pdop = sqrt(Q(0, 0) + Q(1, 1) + Q(2, 2));
    if (pdop == 0 || pdop > 100 || std::isnan(pdop) == true)
    {
        pdop = 100;
    }
    double uz = 0.5 - sin(2*deg2rad(FOV_u))/(4*deg2rad(FOV_u));
    double g_floor = sqrt(4/(1-uz) + (1/uz));
    double R_eff_sq = MEAN_RANGE*MEAN_RANGE - blind*blind;
    double N_typical = 4.0 * M_PI * std::sin(deg2rad(FOV_u)) * R_eff_sq / (DOP_VOXEL_SIZE * DOP_VOXEL_SIZE);
    double rho = pdop * sqrt(N_typical)/g_floor;

    return rho;
}

std::optional<gtsam::Pose3> doGICPVirtualRelative( int _loop_kf_idx, int _curr_kf_idx, Eigen::Matrix4f delta_TF)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr targetKeyframeCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::io::loadPCDFile(dir2_scans_path + std::to_string(_curr_kf_idx) + ".pcd", *cureKeyframeCloud);
    pcl::io::loadPCDFile(dir1_scans_path + std::to_string(_loop_kf_idx) + ".pcd", *targetKeyframeCloud);
    pcl::VoxelGrid<pcl::PointXYZI> downSizeFilter;
    downSizeFilter.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
    downSizeFilter.setInputCloud(cureKeyframeCloud);
    downSizeFilter.filter(*cureKeyframeCloud);
    downSizeFilter.setInputCloud(targetKeyframeCloud);
    downSizeFilter.filter(*targetKeyframeCloud);

    gicp.setInputTarget(targetKeyframeCloud);
    gicp.setInputSource(cureKeyframeCloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    gicp.align(*aligned_cloud, delta_TF);
    Eigen::Matrix4f edge_TF = gicp.getFinalTransformation();
    pcl::PointCloud<pcl::PointXYZI>::Ptr matchKeyframeCloud (new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*cureKeyframeCloud, *matchKeyframeCloud, edge_TF);

    Eigen::Matrix<double, 6, 6> hessian = gicp.getHessian();
    // PD 보장을 위한 최소 정규화 + H 직접 입력
    double lambda = 1e-6 * hessian.diagonal().array().abs().maxCoeff();
    Eigen::Matrix<double, 6, 6> hessian_reg = hessian + lambda * Eigen::Matrix<double, 6, 6>::Identity();

    pcl::PointCloud<pcl::PointXYZI>::Ptr MatchedCloud (new pcl::PointCloud<pcl::PointXYZI>());

    std::unordered_set<int64_t> targetVoxels;
    targetVoxels.reserve(targetKeyframeCloud->points.size());
    for (const auto& pt : targetKeyframeCloud->points)
    targetVoxels.insert(pointToKey(pt, VOXEL_SIZE));

    MatchedCloud->reserve(matchKeyframeCloud->points.size());
    for (const auto& pt : matchKeyframeCloud->points)
    {
        if (targetVoxels.count(pointToKey(pt, VOXEL_SIZE)) != 0)
        MatchedCloud->points.push_back(pt);
    }

    double matching_dop = computeDOP(MatchedCloud, Eigen::Vector3d(edge_TF(0,3),edge_TF(1,3),edge_TF(2,3)));
    double curr_dop = computeDOP(cureKeyframeCloud, Eigen::Vector3d(0,0,0));
    double target_dop = computeDOP(targetKeyframeCloud, Eigen::Vector3d(0,0,0));    
    double max_dop;
    if (curr_dop > target_dop)  max_dop = curr_dop;
    else    max_dop = target_dop;
    
    double dop_ratio = matching_dop / max_dop;

    if (dop_ratio < dop_thres && matching_dop < 1.2)
    {
        Eigen::Matrix3f edge_rot = edge_TF.block(0, 0, 3, 3);
        Eigen::Quaternionf final_q(edge_rot);
        // Get pose transformation
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf2::Quaternion(final_q.x(), final_q.y(), final_q.z(), final_q.w())).getRPY(roll, pitch, yaw);
        gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));
        gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(edge_TF(0,3), edge_TF(1,3), edge_TF(2,3)));
        Eigen::Matrix<double, 6, 1> diag_reg = hessian_reg.diagonal().cwiseInverse();
        robustNoiseVector6 << diag_reg(0), diag_reg(1), diag_reg(2), diag_reg(3), diag_reg(4), diag_reg(5);
        auto info_model = gtsam::noiseModel::Gaussian::Information(hessian_reg);

        robustLoopNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(2.0), info_model);

        isLoopClosed = true;
        return poseFrom.between(poseTo);
    }
    else
    {
        return std::nullopt;
    }
}

void updatePoses(void)
{
    Merge_path.poses.clear();
      // 첫 번째 맵 처리
    for (int i = 0; i < FirstMapSize; i++) 
    {
        int global_key = getGlobalNodeIdx(1, i);
        // 첫 번째 맵 처리 로직
        geometry_msgs::msg::PoseStamped poseStampPGO;
        poseStampPGO.header.frame_id = "map";
        poseStampPGO.pose.position.x = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().x();
        poseStampPGO.pose.position.y = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().y();
        poseStampPGO.pose.position.z = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().z();
        tf2::Quaternion quat_tf2;
        quat_tf2.setRPY(isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().roll(), isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().pitch(), isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().yaw());
        poseStampPGO.pose.orientation = tf2::toMsg(quat_tf2);
        Merge_path.header.frame_id = "map";
        Merge_path.poses.push_back(poseStampPGO);
        
        optimized_stream << FirstMapTime[i] << " "
            << poseStampPGO.pose.position.x << " " << poseStampPGO.pose.position.y << " " << poseStampPGO.pose.position.z << " " 
            << quat_tf2.x() << " " << quat_tf2.y() << " " << quat_tf2.z() << " " << quat_tf2.w() << endl;
    }

    // 두 번째 맵 처리  
    for (int i = 0; i < SecondMapSize; i++) 
    {
        int global_key = getGlobalNodeIdx(2, i);
        // 두 번째 맵 처리 로직
        geometry_msgs::msg::PoseStamped poseStampPGO;
        poseStampPGO.header.frame_id = "map";
        poseStampPGO.pose.position.x = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().x();
        poseStampPGO.pose.position.y = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().y();
        poseStampPGO.pose.position.z = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().z();
        tf2::Quaternion quat_tf2;
        quat_tf2.setRPY(isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().roll(), isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().pitch(), isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().yaw());
        poseStampPGO.pose.orientation = tf2::toMsg(quat_tf2);
        Merge_path.header.frame_id = "map";
        Merge_path.poses.push_back(poseStampPGO);
        
        optimized_stream << SecondMapTime[i] << " "
            << poseStampPGO.pose.position.x << " " << poseStampPGO.pose.position.y << " " << poseStampPGO.pose.position.z << " " 
            << quat_tf2.x() << " " << quat_tf2.y() << " " << quat_tf2.z() << " " << quat_tf2.w() << endl;
    }
    
    PubMerge_path->publish(Merge_path);
}

void runISAM2opt(void)
{
    // called when a variable added
    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isam->update();


    gtSAMgraph.resize(0);
    initialEstimate.clear();

    isamCurrentEstimate = isam->calculateEstimate();
    recentIdxUpdated = int(isamCurrentEstimate.size());
    updatePoses();
}

void generateOptimizedMap()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr FirstOptimizedMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr FirstGroundMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr FirstNonGroundMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> downSizeMapFilter;
    downSizeMapFilter.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);

    for (int i = 0; i < FirstMapSize; i++) 
    {
        int global_key = getGlobalNodeIdx(1, i);
        Pose6 keyPose;
        keyPose.x = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().x();
        keyPose.y = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().y();
        keyPose.z = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().z();
        keyPose.roll = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().roll();
        keyPose.pitch = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().pitch();
        keyPose.yaw = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().yaw();

        MergeMapPoses.push_back(keyPose);        

        Eigen::Matrix4f TF = createTransformMatrix(keyPose);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud = loadPointCloud(dir1_scans_path + to_string(i) + ".pcd");
        pcl::PointCloud<pcl::PointXYZI>::Ptr cureNGKeyframeCloud = loadPointCloud(dir1_scans_path + to_string(i) + "_nonground.pcd");
        pcl::PointCloud<pcl::PointXYZI>::Ptr cureGKeyframeCloud = loadPointCloud(dir1_scans_path + to_string(i) + "_ground.pcd");

        pcl::io::savePCDFileBinary(ScanDirectory + to_string(i) + ".pcd", *cureKeyframeCloud); // scan data 
        pcl::io::savePCDFileBinary(ScanDirectory + to_string(i) + "_nonground.pcd", *cureNGKeyframeCloud); // scan data 
        pcl::io::savePCDFileBinary(ScanDirectory + to_string(i) + "_ground.pcd", *cureGKeyframeCloud); // scan data 
        
        pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud, TF);
        pcl::transformPointCloud(*cureNGKeyframeCloud, *cureNGKeyframeCloud, TF);
        pcl::transformPointCloud(*cureGKeyframeCloud, *cureGKeyframeCloud, TF);
        
        
        std::for_each(cureKeyframeCloud->points.begin(), cureKeyframeCloud->points.end(),
            [](pcl::PointXYZI& point) { point.intensity = 1.0f; });

        *FirstOptimizedMapCloud += *cureKeyframeCloud;       
        *FirstGroundMapCloud += *cureGKeyframeCloud;       
        *FirstNonGroundMapCloud += *cureNGKeyframeCloud;       
    }
    downSizeMapFilter.setInputCloud(FirstOptimizedMapCloud);
    downSizeMapFilter.filter(*FirstOptimizedMapCloud);
    downSizeMapFilter.setInputCloud(FirstGroundMapCloud);
    downSizeMapFilter.filter(*FirstGroundMapCloud);
    downSizeMapFilter.setInputCloud(FirstNonGroundMapCloud);
    downSizeMapFilter.filter(*FirstNonGroundMapCloud);
    pcl::io::savePCDFileBinary(save_directory + "FirstMap.pcd", *FirstOptimizedMapCloud); 
    pcl::io::savePCDFileBinary(save_directory + "FirstGroundMap.pcd", *FirstGroundMapCloud); 
    pcl::io::savePCDFileBinary(save_directory + "FirstNonGroundMap.pcd", *FirstNonGroundMapCloud); 

    pcl::PointCloud<pcl::PointXYZI>::Ptr SecondOptimizedMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr SecondGroundMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr SecondNonGroundMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    for (int i = 0; i < SecondMapSize; i++) 
    {
        int global_key = getGlobalNodeIdx(2, i);
        Pose6 keyPose;
        keyPose.x = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().x();
        keyPose.y = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().y();
        keyPose.z = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().z();
        keyPose.roll = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().roll();
        keyPose.pitch = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().pitch();
        keyPose.yaw = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().yaw();

        MergeMapPoses.push_back(keyPose);

        Eigen::Matrix4f TF = createTransformMatrix(keyPose);
        
        pcl::PointCloud<pcl::PointXYZI>::Ptr cureKeyframeCloud = loadPointCloud(dir2_scans_path + to_string(i) + ".pcd");
        pcl::PointCloud<pcl::PointXYZI>::Ptr cureNGKeyframeCloud = loadPointCloud(dir2_scans_path + to_string(i) + "_nonground.pcd");
        pcl::PointCloud<pcl::PointXYZI>::Ptr cureGKeyframeCloud = loadPointCloud(dir2_scans_path + to_string(i) + "_ground.pcd");
        
        pcl::io::savePCDFileBinary(ScanDirectory + to_string(i+FirstMapSize) + ".pcd", *cureKeyframeCloud); // scan data 
        pcl::io::savePCDFileBinary(ScanDirectory + to_string(i+FirstMapSize) + "_nonground.pcd", *cureNGKeyframeCloud); // scan data 
        pcl::io::savePCDFileBinary(ScanDirectory + to_string(i+FirstMapSize) + "_ground.pcd", *cureGKeyframeCloud); // scan data 

        pcl::transformPointCloud(*cureKeyframeCloud, *cureKeyframeCloud, TF);
        pcl::transformPointCloud(*cureNGKeyframeCloud, *cureNGKeyframeCloud, TF);
        pcl::transformPointCloud(*cureGKeyframeCloud, *cureGKeyframeCloud, TF);
        
        std::for_each(cureKeyframeCloud->points.begin(), cureKeyframeCloud->points.end(),
            [](pcl::PointXYZI& point) { point.intensity = 2.0f; });

        *SecondOptimizedMapCloud += *cureKeyframeCloud;
        *SecondGroundMapCloud += *cureGKeyframeCloud;
        *SecondNonGroundMapCloud += *cureNGKeyframeCloud;
    }

    downSizeMapFilter.setInputCloud(SecondOptimizedMapCloud);
    downSizeMapFilter.filter(*SecondOptimizedMapCloud);
    downSizeMapFilter.setInputCloud(SecondGroundMapCloud);
    downSizeMapFilter.filter(*SecondGroundMapCloud);
    downSizeMapFilter.setInputCloud(SecondNonGroundMapCloud);
    downSizeMapFilter.filter(*SecondNonGroundMapCloud);

    pcl::io::savePCDFileBinary(save_directory + "SecondMap.pcd", *SecondOptimizedMapCloud);
    pcl::io::savePCDFileBinary(save_directory + "SecondGroundMap.pcd", *SecondGroundMapCloud); 
    pcl::io::savePCDFileBinary(save_directory + "SecondNonGroundMap.pcd", *SecondNonGroundMapCloud); 

}

void saveEdge(tuple<int, int, gtsam::Vector, Pose6> edge, int idx)
{
    const int prev_node_idx = get<0>(edge) + idx;
    const int curr_node_idx = get<1>(edge) + idx;
    gtsam::Vector edge_score = get<2>(edge);
    const Pose6 edge_pose = get<3>(edge);
    noiseModel::Diagonal::shared_ptr EdgeNoise = noiseModel::Diagonal::Variances(edge_score);
    gtsam::Pose3 relative_pose = Pose6toGTSAMPose3(edge_pose);
    edge_stream << prev_node_idx << " " << curr_node_idx << " " << relative_pose.translation().x() << " " << relative_pose.translation().y() << " " 
        << relative_pose.translation().z() << " " << relative_pose.rotation().roll() << " " << relative_pose.rotation().pitch() << " " 
        << relative_pose.rotation().yaw() << " " << edge_score(0) << " " << edge_score(1) << " " << edge_score(2) << " " << edge_score(3) << " " 
        << edge_score(4) << " " << edge_score(5) << endl;
}

void getDirectory()
{
    // ROOT_DIR 경로 설정 (현재 프로젝트 디렉토리)
    std::string root_path = string(ROOT_DIR);

    save_directory = root_path + output_directory + "/";
    auto unused = system((std::string("exec rm -r ") + save_directory).c_str());
    unused = system((std::string("mkdir -p ") + save_directory).c_str());

    DebugDirectory = save_directory + "Debug/";
    unused = system((std::string("exec rm -r ") + DebugDirectory).c_str());
    unused = system((std::string("mkdir -p ") + DebugDirectory).c_str());
    
    ScanDirectory = save_directory + "Scans/";
    unused = system((std::string("exec rm -r ") + ScanDirectory).c_str());
    unused = system((std::string("mkdir -p ") + ScanDirectory).c_str());
    
    string optimized_path = save_directory + "/optimized_poses.txt";
    optimized_stream = std::fstream(optimized_path, std::fstream::out);
    optimized_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!optimized_stream) 
    {
        cout<<"Failed to open graph optimization file"<<endl;
    }

    string edge_directory = save_directory + "/edges.txt";
    edge_stream = std::fstream(edge_directory, std::fstream::out);
    edge_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!edge_stream) 
    {
        cout<< "Failed to open Edge file"<<endl;
    }
    
    // 첫 번째 디렉토리 경로들
    dir1_scans_path = directory1 + "/Scans/";
    dir1_poses_path = directory1 + "/optimized_poses.txt";
    dir1_edges_path = directory1 + "/edges.txt";
    dir1_map_path   = directory1 + "/StaticMap.pcd";
    
    // 두 번째 디렉토리 경로들
    dir2_scans_path = directory2 + "/Scans/";
    dir2_poses_path = directory2 + "/optimized_poses.txt";
    dir2_edges_path = directory2 + "/edges.txt";
    dir2_map_path   = directory2 + "/StaticMap.pcd";
    
    // 경로 출력으로 확인
    std::cout << "=== Directory Paths ===" << std::endl;
    std::cout << "Directory 1 Scans: " << dir1_scans_path << std::endl;
    std::cout << "Directory 1 Poses: " << dir1_poses_path << std::endl;
    std::cout << "Directory 1 Edges: " << dir1_edges_path << std::endl;
    std::cout << "Directory 2 Scans: " << dir2_scans_path << std::endl;
    std::cout << "Directory 2 Poses: " << dir2_poses_path << std::endl;
    std::cout << "Directory 2 Edges: " << dir2_edges_path << std::endl;
    std::cout << "======================" << std::endl;
    
}

// 포즈 파일을 읽는 함수
bool loadPoseFile(const std::string& filepath, std::vector<double>& times, std::vector<Pose6>& poses)
{
    std::ifstream file(filepath);
    if (!file.is_open()) 
    {
        std::cerr << "Error: Could not open pose file: " << filepath << std::endl;
        return false;
    }
    
    std::string line;
    int line_count = 0;
    
    while (std::getline(file, line)) 
    {
        line_count++;
        if (line.empty()) continue;  // 빈 줄 건너뛰기
        
        std::istringstream iss(line);
        double time, x, y, z, qx, qy, qz, qw;
        
        // 한 줄에서 8개 값을 읽기: time x y z qx qy qz qw
        if (!(iss >> time >> x >> y >> z >> qx >> qy >> qz >> qw)) 
        {
            std::cerr << "Error: Failed to parse line " << line_count << " in file: " << filepath << std::endl;
            std::cerr << "Line content: " << line << std::endl;
            continue;  // 파싱 실패한 줄은 건너뛰고 계속 진행
        }
        
        // 시간을 벡터에 저장
        times.push_back(time);
        
        // Euler angles로 변환하여 Pose6로 저장
        Pose6 pose = poseToPose6(x, y, z, qx, qy, qz, qw);
        poses.push_back(pose);
    }
    
    file.close();
    
    std::cout << "Successfully loaded " << times.size() << " poses from: " << filepath << std::endl;
    return true;
}

// 엣지 파일을 읽는 함수
bool loadEdgeFile(const std::string& filepath, std::vector<tuple<int, int, gtsam::Vector, Pose6>>& edges)
{
    std::ifstream file(filepath);
    if (!file.is_open()) 
    {
        std::cerr << "Error: Could not open edge file: " << filepath << std::endl;
        return false;
    }
    
    std::string line;
    int line_count = 0;
    
    while (std::getline(file, line)) 
    {
        line_count++;
        if (line.empty()) continue;  // 빈 줄 건너뛰기
        
        std::istringstream iss(line);
        int prev_idx, curr_idx;
        double covaricance1, covaricance2, covaricance3, covaricance4, covaricance5, covaricance6;
        Pose6 pose;
        
        // 한 줄에서 8개 값을 읽기: time x y z qx qy qz qw
        if (!(iss >> prev_idx >> curr_idx >> pose.x >> pose.y >> pose.z >> pose.roll >> pose.pitch >> pose.yaw
                >> covaricance1 >> covaricance2 >> covaricance3 >> covaricance4 >> covaricance5 >> covaricance6)) 
        {
            std::cerr << "Error: Failed to parse line " << line_count << " in file: " << filepath << std::endl;
            std::cerr << "Line content: " << line << std::endl;
            continue;  // 파싱 실패한 줄은 건너뛰고 계속 진행
        }
        gtsam::Vector edgeNoiseVector(6);
        edgeNoiseVector << covaricance1, covaricance2, covaricance3, covaricance4, covaricance5, covaricance6;
        tuple<int, int, gtsam::Vector, Pose6> edge = make_tuple(prev_idx, curr_idx, edgeNoiseVector, pose);
        
        edges.push_back(edge);
    }
    
    file.close();
    
    std::cout << "Successfully loaded " << edges.size() << " edges from: " << filepath << std::endl;
    return true;
}

bool loadFiles()
{
    // 포즈 파일들 읽기
    std::cout << "Loading pose files..." << std::endl;

    // 첫 번째 맵의 포즈 파일 읽기
    if (!loadPoseFile(dir1_poses_path, FirstMapTime, FirstMapPoses)) 
    {
        std::cerr << "Failed to load first map poses!" << std::endl;
        return false;
    }
    
    // 첫 번째 맵의 엣지 파일 읽기
    if (!loadEdgeFile(dir1_edges_path, FirstMapEdges)) 
    {
        std::cerr << "Failed to load first map edges!" << std::endl;
        return false;
    }

    // 두 번째 맵의 포즈 파일 읽기
    if (!loadPoseFile(dir2_poses_path, SecondMapTime, SecondMapPoses)) 
    {
        std::cerr << "Failed to load second map poses!" << std::endl;
        return false;
    }
    
    // 두 번째 맵의 엣지 파일 읽기
    if (!loadEdgeFile(dir2_edges_path, SecondMapEdges)) 
    {
        std::cerr << "Failed to load second map edges!" << std::endl;
        return false;
    }

    FirstMapSize = FirstMapPoses.size();
    SecondMapSize = SecondMapPoses.size();
    return true;
}

void getEdges()
{
    for (int k = 0; k < FirstMapEdges.size(); k++)
    {
        auto edge = FirstMapEdges[k];
        const int prev_node_idx = get<0>(edge);
        const int curr_node_idx = get<1>(edge);
        gtsam::Vector edge_score = get<2>(edge);
        const Pose6 edge_pose = get<3>(edge);
        noiseModel::Diagonal::shared_ptr EdgeNoise = noiseModel::Diagonal::Variances(edge_score);
        gtsam::Pose3 relative_pose = Pose6toGTSAMPose3(edge_pose);
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(getGlobalNodeIdx(1,prev_node_idx), getGlobalNodeIdx(1,curr_node_idx), relative_pose, EdgeNoise));
        
    }
        
    for (int k = 0; k < SecondMapEdges.size(); k++)
    {
        auto edge = SecondMapEdges[k];
        const int prev_node_idx = get<0>(edge);
        const int curr_node_idx = get<1>(edge);
        gtsam::Vector edge_score = get<2>(edge);
        const Pose6 edge_pose = get<3>(edge);
        noiseModel::Diagonal::shared_ptr EdgeNoise = noiseModel::Diagonal::Variances(edge_score);
        gtsam::Pose3 relative_pose = Pose6toGTSAMPose3(edge_pose);
        gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(getGlobalNodeIdx(2,prev_node_idx), getGlobalNodeIdx(2,curr_node_idx), relative_pose, EdgeNoise));
    }
}

void placeRecognition()
{
    std::cout << "\033[2J\033[H";  // Clear screen and move cursor to top
    std::cout << "First Map SOLiD Descriptor generating..." << std::endl;
    for (int k = 0; k < FirstMapSize; k++)
    {             
        pcl::PointCloud<pcl::PointXYZI>::Ptr curr_pc (new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr curr_pc_down (new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(dir1_scans_path + std::to_string(k) + ".pcd", *curr_pc);
        solidModule.down_sampling(*curr_pc, curr_pc_down);
        solidModule.makeAndSaveSolid(*curr_pc_down);       
    }
    
    std::cout << "\033[2J\033[H";  // Clear screen and move cursor to top
    std::cout << "Second Map SOLiD Descriptor generating..." << std::endl;
    for (int k = 0; k < SecondMapSize; k++)
    {             
        pcl::PointCloud<pcl::PointXYZI>::Ptr curr_pc (new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr curr_pc_down (new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(dir2_scans_path + std::to_string(k) + ".pcd", *curr_pc);
        solidModule.down_sampling(*curr_pc, curr_pc_down);
        solidModule.makeAndSaveSolid(*curr_pc_down);       
    }
}

void getLoopEdges()
{    
    pcl::PointCloud<pcl::PointXYZI>::Ptr FirstMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::io::loadPCDFile(dir1_map_path, *FirstMapCloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr SecondMapCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::io::loadPCDFile(dir2_map_path, *SecondMapCloud);
    std::vector<int> first_indices;
    std::vector<int> second_indices;
    pcl::removeNaNFromPointCloud(*FirstMapCloud, *FirstMapCloud, first_indices);
    pcl::removeNaNFromPointCloud(*SecondMapCloud, *SecondMapCloud, second_indices);
    kiss_matcher::KISSMatcherConfig config = kiss_matcher::KISSMatcherConfig(anchor_resolution);
    kiss_matcher::KISSMatcher matcher(config);  

    const auto& tgt_vec = convertCloudToVec(*FirstMapCloud);
    const auto& src_vec = convertCloudToVec(*SecondMapCloud);

    const auto solution = matcher.estimate(src_vec, tgt_vec);
    Eigen::Matrix4d solution_eigen      = Eigen::Matrix4d::Identity();
    solution_eigen.block<3, 3>(0, 0)    = solution.rotation;
    solution_eigen.topRightCorner(3, 1) = solution.translation;    
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr MatchedCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr FirstMatchedCloud(new pcl::PointCloud<pcl::PointXYZI>(*FirstMapCloud));
    pcl::PointCloud<pcl::PointXYZI>::Ptr SecondMatchedCloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*SecondMapCloud, *SecondMatchedCloud, solution_eigen.cast<float>());
    std::for_each(FirstMatchedCloud->points.begin(), FirstMatchedCloud->points.end(),
                [](pcl::PointXYZI& point) { point.intensity = 1.0; });
    std::for_each(SecondMatchedCloud->points.begin(), SecondMatchedCloud->points.end(),
                [](pcl::PointXYZI& point) { point.intensity = 2.0; });
    *MatchedCloud += *FirstMatchedCloud;
    *MatchedCloud += *SecondMatchedCloud;
    pcl::io::savePCDFileBinary(DebugDirectory + "KissMatchedMap" + ".pcd", *MatchedCloud); 

    A2_anchor = gtsam::Pose3(solution_eigen);

    int solidSize = solidModule.getSolidSize();
    for (int k = FirstMapSize; k < solidSize; k++)
    {         
        auto detectResult = solidModule.detectLoopClosureID(k); // first: nn index, second: yaw diff 
        int SOLiDclosestHistoryFrameID = std::get<1>(detectResult);
        if( SOLiDclosestHistoryFrameID != -1 && SOLiDclosestHistoryFrameID < FirstMapSize) 
        {
            const int prev_node_idx = std::get<1>(detectResult);
            const int curr_node_idx = std::get<0>(detectResult)-FirstMapSize; // 첫번째 맵 디스크립터 등록 후이기 때문에 첫번째 맵 포즈 사이즈만큼 뺌

            pcl::PointCloud<pcl::PointXYZI>::Ptr FirstScanCloud(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::io::loadPCDFile(dir1_scans_path + std::to_string(prev_node_idx) + ".pcd", *FirstScanCloud);
            pcl::PointCloud<pcl::PointXYZI>::Ptr SecondScanCloud(new pcl::PointCloud<pcl::PointXYZI>());
            pcl::io::loadPCDFile(dir2_scans_path + std::to_string(curr_node_idx) + ".pcd", *SecondScanCloud);
            std::vector<int> first_indices;
            std::vector<int> second_indices;
            pcl::removeNaNFromPointCloud(*FirstScanCloud, *FirstScanCloud, first_indices);
            pcl::removeNaNFromPointCloud(*SecondScanCloud, *SecondScanCloud, second_indices);
            kiss_matcher::KISSMatcherConfig config = kiss_matcher::KISSMatcherConfig(VOXEL_SIZE);
            kiss_matcher::KISSMatcher matcher(config);  

            const auto& tgt_vec = convertCloudToVec(*FirstScanCloud);
            const auto& src_vec = convertCloudToVec(*SecondScanCloud);
        
            const auto solution = matcher.estimate(src_vec, tgt_vec);
            
            Eigen::Matrix4f delta_TF (Eigen::Matrix4f::Identity());
            delta_TF.block<3, 3>(0, 0)    = solution.rotation.cast<float>();
            delta_TF.topRightCorner(3, 1) = solution.translation.cast<float>();  
            pcl::transformPointCloud(*SecondScanCloud, *SecondScanCloud, delta_TF);            

            // FirstScanCloud 기반 voxel 셋 구성
            std::unordered_set<int64_t> firstScanVoxels;
            firstScanVoxels.reserve(FirstScanCloud->points.size());
            for (const auto& pt : FirstScanCloud->points)
                firstScanVoxels.insert(pointToKey(pt, VOXEL_SIZE));

            // 변환된 SecondScanCloud에서 FirstScanCloud와 매칭되는 점만 추출
            pcl::PointCloud<pcl::PointXYZI>::Ptr MatchingCloud(new pcl::PointCloud<pcl::PointXYZI>());
            MatchingCloud->reserve(SecondScanCloud->points.size());
            for (const auto& pt : SecondScanCloud->points)
            {
                if (firstScanVoxels.count(pointToKey(pt, VOXEL_SIZE)) != 0)
                    MatchingCloud->points.push_back(pt);
            }
            Eigen::Vector3d delta_vec(delta_TF(0,3), delta_TF(1,3), delta_TF(2,3));
            double matching_dop = computeDOP(MatchingCloud, delta_vec);
            if (matching_dop > 0.5)
            {                
                continue;
            }

            auto relative_pose_optional = doGICPVirtualRelative(prev_node_idx, curr_node_idx, delta_TF);
        
            if(relative_pose_optional)
            {
                gtsam::Pose3 relative_pose = relative_pose_optional.value();
                gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                    getGlobalNodeIdx(1, prev_node_idx), getGlobalNodeIdx(2, curr_node_idx), relative_pose, robustLoopNoise));

                edge_stream << std::get<1>(detectResult) << " " << std::get<0>(detectResult) << " " << relative_pose.translation().x() << " " << relative_pose.translation().y() << " " 
                    << relative_pose.translation().z() << " " << relative_pose.rotation().roll() << " " << relative_pose.rotation().pitch() << " " 
                    << relative_pose.rotation().yaw() << " " << robustNoiseVector6(0) << " " << robustNoiseVector6(1) << " " << robustNoiseVector6(2) << " " << robustNoiseVector6(3) << " " 
                    << robustNoiseVector6(4) << " " << robustNoiseVector6(5) << endl;
                pair<int, int> loop_pair = make_pair(prev_node_idx, curr_node_idx);
                loop_pairs.push_back(loop_pair);
            }    
        }       
    }
}

void getPoses()
{
    for (int k = 0; k < FirstMapSize; k++)
    {        
        Pose6 current_pose = FirstMapPoses[k];
        geometry_msgs::msg::PoseStamped poseStamped;
        poseStamped.header.frame_id = "map";
        poseStamped.pose.position.x = current_pose.x;
        poseStamped.pose.position.y = current_pose.y;
        poseStamped.pose.position.z = current_pose.z;
        tf2::Quaternion quat_tf;
        quat_tf.setRPY(current_pose.roll, current_pose.pitch, current_pose.yaw);

        if(!gtSAMgraphMade)
        {
            const int init_node_idx = 0;
            gtsam::Pose3 poseOrigin = Pose6toGTSAMPose3(current_pose);

            // prior factor
            gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(getGlobalNodeIdx(1,k), poseOrigin, priorNoise));
            initialEstimate.insert(getGlobalNodeIdx(1,k), poseOrigin);

            gtSAMgraphMade = true;
        }
        else
        {
            gtsam::Pose3 poseFrom = Pose6toGTSAMPose3(FirstMapPoses.at(k-1));
            gtsam::Pose3 poseTo = Pose6toGTSAMPose3(FirstMapPoses.at(k));
            // odom factor
            gtsam::Pose3 relPose = poseFrom.between(poseTo);

            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(getGlobalNodeIdx(1,k-1), getGlobalNodeIdx(1,k), relPose, odomNoise));
            initialEstimate.insert(getGlobalNodeIdx(1,k), poseTo);        
        } 

        pcl::PointXYZI kf_node;
        kf_node.x = poseStamped.pose.position.x;
        kf_node.y = poseStamped.pose.position.y;
        kf_node.z = poseStamped.pose.position.z;
        kf_node.intensity = k;
        FirstMap_nodes.push_back(kf_node);

        sensor_msgs::msg::PointCloud2 kf_nodes_msg;
        pcl::toROSMsg(FirstMap_nodes, kf_nodes_msg);
        FirstMap_nodes.header.frame_id = "map";
        First_kf_node_pub->publish(kf_nodes_msg);  

        poseStamped.pose.orientation = tf2::toMsg(quat_tf);
        FirstMap_path.header.frame_id = "map";
        FirstMap_path.poses.push_back(poseStamped);
        FirstMap_path.poses[k].header.stamp = poseStamped.header.stamp;
        PubFirstMap_path->publish(FirstMap_path);    
    }
    
    for (int k = 0; k < SecondMapSize; k++)
    {        
        Pose6 current_pose = SecondMapPoses[k];
        gtsam::Pose3 poseCurr = Pose6toGTSAMPose3(current_pose);
        Eigen::Matrix4d curr_TF = poseCurr.matrix();
        Eigen::Matrix4d anchor_TF = A2_anchor.matrix();
        Eigen::Matrix4d anchor_curr_TF = anchor_TF * curr_TF;
        gtsam::Pose3 poseAnchorCurr(anchor_curr_TF);

        geometry_msgs::msg::PoseStamped poseStamped;
        poseStamped.header.frame_id = "map";
        poseStamped.pose.position.x = poseAnchorCurr.translation().x();
        poseStamped.pose.position.y = poseAnchorCurr.translation().y();
        poseStamped.pose.position.z = poseAnchorCurr.translation().z();

        tf2::Quaternion quat_tf;
        quat_tf.setRPY(poseAnchorCurr.rotation().roll(), poseAnchorCurr.rotation().pitch(), poseAnchorCurr.rotation().yaw());
        poseStamped.pose.orientation = tf2::toMsg(quat_tf);

        if (k == 0)
        {
            // prior factor
            gtSAMgraph.add(gtsam::PriorFactor<gtsam::Pose3>(getGlobalNodeIdx(2,k), poseAnchorCurr, largeNoise));
            initialEstimate.insert(getGlobalNodeIdx(2,k), poseAnchorCurr);
        }
        else
        {
            gtsam::Pose3 poseFrom = Pose6toGTSAMPose3(SecondMapPoses.at(k-1));
            gtsam::Pose3 poseTo = Pose6toGTSAMPose3(SecondMapPoses.at(k));
            // odom factor
            gtsam::Pose3 relPose = poseFrom.between(poseTo);

            int current_idx = FirstMapSize + k;
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(getGlobalNodeIdx(2,k-1), getGlobalNodeIdx(2,k), relPose, odomNoise));
            initialEstimate.insert(getGlobalNodeIdx(2,k), poseAnchorCurr);        
        }    

        pcl::PointXYZI kf_node;
        kf_node.x = poseStamped.pose.position.x;
        kf_node.y = poseStamped.pose.position.y;
        kf_node.z = poseStamped.pose.position.z;
        kf_node.intensity = k;
        SecondMap_nodes.push_back(kf_node);

        sensor_msgs::msg::PointCloud2 kf_nodes_msg;
        pcl::toROSMsg(SecondMap_nodes, kf_nodes_msg);
        SecondMap_nodes.header.frame_id = "map";
        Second_kf_node_pub->publish(kf_nodes_msg);  

        SecondMap_path.header.frame_id = "map";
        SecondMap_path.poses.push_back(poseStamped);
        SecondMap_path.poses[k].header.stamp = poseStamped.header.stamp;
        PubSecondMap_path->publish(SecondMap_path);
    }

    for (int i = 0; i < loop_pairs.size(); i++)
    {
        int idx1 = loop_pairs[i].first;
        int idx2 = loop_pairs[i].second;
        geometry_msgs::msg::Point p;
        p.x = FirstMap_nodes[idx1].x;    p.y = FirstMap_nodes[idx1].y;    p.z = FirstMap_nodes[idx1].z;
        loopLine.points.push_back(p);
        p.x = SecondMap_nodes[idx2].x;    p.y = SecondMap_nodes[idx2].y;    p.z = SecondMap_nodes[idx2].z;
        loopLine.points.push_back(p);
        LoopLineMarker_pub->publish(loopLine); 
    }
}


namespace {

using UfoVoidMap = ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION>;

struct FrameData {
    int kf_id;
    Eigen::Vector3f origin;
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_cloud;
};

struct SessionMaps {
    UfoVoidMap map;
    std::vector<FrameData> frames;

    SessionMaps(float res)
        : map(static_cast<ufo::node_size_t>(res), static_cast<ufo::depth_t>(17))
    {
    }
};

struct ClassificationResult {
    pcl::PointCloud<pcl::PointXYZI>::Ptr first_ue{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::PointCloud<pcl::PointXYZI>::Ptr second_ue{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::PointCloud<pcl::PointXYZI>::Ptr nd_cloud{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::PointCloud<pcl::PointXYZI>::Ptr pd_cloud{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::PointCloud<pcl::PointXYZI>::Ptr persistence_cloud{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::PointCloud<pcl::PointXYZI>::Ptr evidence_debug_cloud{new pcl::PointCloud<pcl::PointXYZI>()};
    std::unordered_set<int64_t> first_ue_voxels;
    std::unordered_set<int64_t> second_ue_voxels;
    std::unordered_set<int64_t> nd_voxels;
    std::unordered_set<int64_t> pd_voxels;
    // 후처리 필터 이후에도 "ND/PD yield" 를 다시 계산할 수 있도록 분모를 함께 내보낸다.
    std::size_t void2_total = 0;  // pers_1 에서 2세션이 free 로 관측한 voxel 수 (ND 의 분모)
    std::size_t void1_total = 0;  // pers_2 에서 1세션이 free 로 관측한 voxel 수 (PD 의 분모)
};

inline int64_t voxelKeyFromIndex(int ix, int iy, int iz)
{
    int64_t x = static_cast<int64_t>(ix) & 0x1FFFFF;
    int64_t y = static_cast<int64_t>(iy) & 0x1FFFFF;
    int64_t z = static_cast<int64_t>(iz) & 0x1FFFFF;
    return x | (y << 21) | (z << 42);
}

inline Pose6 getEstimatedPose(int session_id, int local_idx)
{
    int global_key = getGlobalNodeIdx(session_id, local_idx);
    Pose6 key_pose;
    key_pose.x = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().x();
    key_pose.y = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().y();
    key_pose.z = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().z();
    key_pose.roll = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().roll();
    key_pose.pitch = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().pitch();
    key_pose.yaw = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().yaw();
    return key_pose;
}

inline pcl::PointCloud<pcl::PointXYZI>::Ptr loadWorldScan(int session_id, int local_idx,
                                                          Eigen::Vector3f& origin)
{
    Pose6 key_pose = getEstimatedPose(session_id, local_idx);
    Eigen::Matrix4f tf = createTransformMatrix(key_pose);
    origin = Eigen::Vector3f(tf(0, 3), tf(1, 3), tf(2, 3));

    std::string scan_path = (session_id == 1 ? dir1_scans_path : dir2_scans_path) +
                            std::to_string(local_idx) + ".pcd";
    auto cloud = loadPointCloud(scan_path);
    if (!cloud) {
        return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    }

    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, idx);
    pcl::transformPointCloud(*cloud, *cloud, tf);
    return cloud;
}

inline ufo::PointCloud toUfoCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud)
{
    ufo::PointCloud out;
    out.reserve(cloud.points.size());
    for (const auto& p : cloud.points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        out.emplace_back(ufo::Point(p.x, p.y, p.z));
    }
    return out;
}

SessionMaps buildVoidMap(int session_id, int map_size, const lt_mapping::EvidenceParams& params)
{
    SessionMaps session_map(params.res);
    session_map.map.reserve(30'000'000);
    session_map.frames.reserve(static_cast<size_t>(map_size));

    ufo::IntegrationParams integ;
    integ.min_range = 1.0f;
    integ.max_range = static_cast<float>(MAX_DISTANCE);
    integ.inflate_hits_dist = 0.2f;
    integ.inflate_unknown = 1;
    integ.ray_passthrough_hits = true;
    integ.parallel = true;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < map_size; ++i) {
        Eigen::Vector3f origin;
        auto world_cloud = loadWorldScan(session_id, i, origin);

        FrameData frame;
        frame.kf_id = i;
        frame.origin = origin;
        frame.world_cloud = world_cloud;
        session_map.frames.push_back(frame);

        ufo::PointCloud ucloud = toUfoCloud(*world_cloud);
        ufo::insertPointCloud(session_map.map, ucloud, ufo::Point(origin.x(), origin.y(), origin.z()),
                              integ, false);
    }
    session_map.map.propagateModified();

    const auto t1 = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Session %d void map insertion finished in %.3f sec (%d keyframes)",
                session_id, static_cast<double>(ms) / 1000.0, map_size);

    return session_map;
}

std::unordered_set<int64_t> buildInterestSet(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                             float res)
{
    std::unordered_set<int64_t> keys;
    keys.reserve(cloud->points.size());
    for (const auto& p : cloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        keys.insert(lt_mapping::voxelKey(p.x, p.y, p.z, res));
    }
    return keys;
}

inline void updateVoidEvidence(std::unordered_map<int64_t, lt_mapping::VoidEvidence>& ev_map,
                               int64_t key, float q_i, const Eigen::Vector3f& u, int kf_id,
                               std::array<std::mutex, 64>& shard_mutex)
{
    std::size_t shard = static_cast<std::size_t>(key) & 63U;
    std::lock_guard<std::mutex> lock(shard_mutex[shard]);
    // 삽입 금지: 엔트리는 accumulateEvidence()의 직렬 프리패스에서 이미 생성되어 있다.
    // 병렬 구간에서 버킷 체인/size를 건드리면 샤드 뮤텍스로는 보호되지 않는다.
    auto it = ev_map.find(key);
    if (it == ev_map.end()) {
        return;
    }
    auto& ev = it->second;
    ev.M += q_i * (u * u.transpose());
    ev.q_sum += q_i;
    if (ev.n_rays < std::numeric_limits<uint16_t>::max()) {
        ev.n_rays++;
    }
    if (ev.last_kf != kf_id) {
        if (ev.n_keyframes < std::numeric_limits<uint16_t>::max()) {
            ev.n_keyframes++;
        }
        ev.last_kf = kf_id;
    }
}

void accumulateRayEvidence(const Eigen::Vector3f& origin, const pcl::PointXYZI& endpoint,
                           const Eigen::Vector3f& normal, int kf_id,
                           const std::unordered_set<int64_t>& interest_set,
                           std::unordered_map<int64_t, lt_mapping::VoidEvidence>& ev_map,
                           const lt_mapping::EvidenceParams& params,
                           std::array<std::mutex, 64>& shard_mutex)
{
    Eigen::Vector3f end(endpoint.x, endpoint.y, endpoint.z);
    Eigen::Vector3f ray = end - origin;
    float ray_len = ray.norm();
    if (ray_len <= std::max(params.res, 1e-6f) || ray_len <= 1.0f) {
        return;
    }

    Eigen::Vector3f u = ray / ray_len;
    const float start_t = 1.0f;
    // UFOMap 삽입(integ.max_range)과 hit 필터링이 모두 MAX_DISTANCE 에서 잘리므로 그 밖의 voxel 은
    // 양쪽 맵에서 unknown 이다. DDA 도 같은 사거리에서 끊어 무의미한 증거 누적을 없앤다.
    const float end_t = std::min(ray_len, static_cast<float>(MAX_DISTANCE));
    if (start_t >= end_t) {
        return;
    }

    Eigen::Vector3f start = origin + u * start_t;
    const float usable_len = end_t - start_t;

    int ix = static_cast<int>(std::floor(start.x() / params.res));
    int iy = static_cast<int>(std::floor(start.y() / params.res));
    int iz = static_cast<int>(std::floor(start.z() / params.res));
    int tx = static_cast<int>(std::floor(end.x() / params.res));
    int ty = static_cast<int>(std::floor(end.y() / params.res));
    int tz = static_cast<int>(std::floor(end.z() / params.res));

    auto init_axis = [&](float s, float ds, int i, float& t_max, float& t_delta, int& step) {
        if (std::abs(ds) < 1e-8f) {
            step = 0;
            t_max = std::numeric_limits<float>::infinity();
            t_delta = std::numeric_limits<float>::infinity();
            return;
        }
        step = (ds > 0.0f) ? 1 : -1;
        float next_boundary = (step > 0) ? ((i + 1) * params.res) : (i * params.res);
        t_max = (next_boundary - s) / ds;
        if (t_max < 0.0f) {
            t_max = 0.0f;
        }
        t_delta = params.res / std::abs(ds);
    };

    float t_max_x, t_max_y, t_max_z;
    float t_delta_x, t_delta_y, t_delta_z;
    int step_x, step_y, step_z;
    init_axis(start.x(), u.x(), ix, t_max_x, t_delta_x, step_x);
    init_axis(start.y(), u.y(), iy, t_max_y, t_delta_y, step_y);
    init_axis(start.z(), u.z(), iz, t_max_z, t_delta_z, step_z);

    const float q_ang_const =
        (normal.allFinite() && normal.norm() > 1e-6f) ? std::abs(u.dot(normal.normalized())) : 1.0f;

    for (;;) {
        if (!(ix == tx && iy == ty && iz == tz)) {
            int64_t key = voxelKeyFromIndex(ix, iy, iz);
            if (interest_set.find(key) != interest_set.end()) {
                Eigen::Vector3f center((static_cast<float>(ix) + 0.5f) * params.res,
                                       (static_cast<float>(iy) + 0.5f) * params.res,
                                       (static_cast<float>(iz) + 0.5f) * params.res);
                float r_i = (center - origin).norm();
                float q_dist = std::exp(-std::max(0.0f, r_i - params.r0) / std::max(params.r_s, 1e-3f));
                float s_i = (end - center).norm();
                float q_ang = (s_i < params.d_surf) ? q_ang_const : 1.0f;
                float q_i = std::max(params.w_min, q_dist * q_ang);
                updateVoidEvidence(ev_map, key, q_i, u, kf_id, shard_mutex);
            }
        } else {
            break;
        }

        float next_t = std::min(t_max_x, std::min(t_max_y, t_max_z));
        if (next_t > usable_len) {
            break;
        }

        if (t_max_x <= t_max_y && t_max_x <= t_max_z) {
            ix += step_x;
            t_max_x += t_delta_x;
        } else if (t_max_y <= t_max_x && t_max_y <= t_max_z) {
            iy += step_y;
            t_max_y += t_delta_y;
        } else {
            iz += step_z;
            t_max_z += t_delta_z;
        }
    }
}

void estimateNormalsForFrame(const FrameData& frame, std::vector<Eigen::Vector3f>& normals)
{
    normals.assign(frame.world_cloud->points.size(), Eigen::Vector3f::UnitZ());
    if (frame.world_cloud->empty()) {
        return;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr ds_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::VoxelGrid<pcl::PointXYZI> vg;
    vg.setLeafSize(0.4f, 0.4f, 0.4f);
    vg.setInputCloud(frame.world_cloud);
    vg.filter(*ds_cloud);
    if (ds_cloud->empty()) {
        return;
    }

    pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::Normal> ne;
    ne.setInputCloud(ds_cloud);
    ne.setRadiusSearch(0.5);
    ne.setViewPoint(frame.origin.x(), frame.origin.y(), frame.origin.z());
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr ds_normals(new pcl::PointCloud<pcl::Normal>());
    ne.compute(*ds_normals);

    pcl::KdTreeFLANN<pcl::PointXYZI> nn_tree;
    nn_tree.setInputCloud(ds_cloud);

    std::vector<int> nn_idx(1);
    std::vector<float> nn_sqdist(1);
    for (size_t i = 0; i < frame.world_cloud->points.size(); ++i) {
        if (nn_tree.nearestKSearch(frame.world_cloud->points[i], 1, nn_idx, nn_sqdist) > 0) {
            const auto& n = ds_normals->points[nn_idx[0]];
            if (std::isfinite(n.normal_x) && std::isfinite(n.normal_y) && std::isfinite(n.normal_z)) {
                Eigen::Vector3f nv(n.normal_x, n.normal_y, n.normal_z);
                if (nv.norm() > 1e-6f) {
                    normals[i] = nv.normalized();
                }
            }
        }
    }
}

void accumulateEvidence(const std::vector<FrameData>& frames,
                        const std::unordered_set<int64_t>& interest_set,
                        std::unordered_map<int64_t, lt_mapping::VoidEvidence>& ev_map,
                        const lt_mapping::EvidenceParams& params)
{
    ev_map.clear();
    ev_map.reserve(interest_set.size());
    // 프리패스: 관심 voxel 엔트리를 직렬로 모두 생성해 둔다. 이후 병렬 구간은 기존 엔트리의
    // 값만 갱신하므로 rehash/버킷 체인/size 변경이 발생하지 않는다.
    for (int64_t key : interest_set) {
        ev_map.emplace(key, lt_mapping::VoidEvidence{});
    }
    std::array<std::mutex, 64> shard_mutex;

    const auto t_begin = std::chrono::steady_clock::now();
    for (const auto& frame : frames) {
        std::vector<Eigen::Vector3f> normals;
        estimateNormalsForFrame(frame, normals);

#pragma omp parallel for schedule(dynamic, 128)
        for (int i = 0; i < static_cast<int>(frame.world_cloud->points.size()); ++i) {
            const auto& p = frame.world_cloud->points[static_cast<size_t>(i)];
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                continue;
            }
            accumulateRayEvidence(frame.origin, p, normals[static_cast<size_t>(i)], frame.kf_id,
                                  interest_set, ev_map, params, shard_mutex);
        }
    }
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_begin).count();
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Evidence accumulation: %zu frames, %zu interest voxels, dda_max_range=%d m, "
                "elapsed=%.2f s",
                frames.size(), interest_set.size(), MAX_DISTANCE, elapsed_s);
}

inline float clampLogOdds(float l, float lim)
{
    return std::max(-lim, std::min(lim, l));
}

struct WeightComponents {
    float w_v;
    float w_geom;         // 신규: lambda_2/lambda_1 기반 방향 다양성
    float aniso;          // lambda_2/lambda_1 원값 (진단용)
    float vdop;           // legacy 3D vDOP (진단 비교용, 판정 미사용)
    float w_geom_legacy;  // legacy vDOP 기반 w_geom (진단 비교용, 판정 미사용)
    float w_obs;
};

WeightComponents computeVoidWeight(const lt_mapping::VoidEvidence& ev,
                                   const lt_mapping::EvidenceParams& params)
{
    if (ev.q_sum <= 0.0f) {
        return {params.w_min, params.w_min, 0.0f, -1.0f, params.w_min, 0.0f};
    }

    // M 을 q_sum(=trace M)으로 정규화해 관측량 스케일을 제거한 뒤 고유값을 구한다.
    // 지상 로봇이 평면 경로를 주행하며 한 voxel 을 훑으면 ray 방향이 거의 한 평면에 놓이므로
    // lambda_3 는 물리적으로 항상 0 근처다(실측 lambda_min/lambda_max 중앙값 1e-4~1e-3).
    // 따라서 lambda_3 에 의존하는 3D vDOP 는 변별력이 없고, 관측 가능한 평면 안에서의
    // 방향 확산 lambda_2/lambda_1 만으로 방향 다양성을 평가한다.
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
    es.computeDirect(ev.M / std::max(ev.q_sum, 1e-6f), Eigen::EigenvaluesOnly);
    const auto& lam = es.eigenvalues();  // Eigen 은 오름차순: (0)=lambda_3, (1)=lambda_2, (2)=lambda_1
    float lam1 = std::max(lam(2), 1e-9f);
    float lam2 = std::max(lam(1), 0.0f);
    float aniso = std::min(lam2 / lam1, 1.0f);
    float w_geom = std::clamp(aniso / std::max(params.aniso_typ, 1e-3f), 0.0f, 1.0f);

    // 아래 legacy 3D vDOP 는 판정에 쓰지 않는다. 신·구 지표를 같은 실행에서 대조하기 위한 진단값.
    Eigen::Matrix3f m_reg = ev.M + params.eps_reg * Eigen::Matrix3f::Identity();
    float vdop = 1e6f;
    float det = m_reg.determinant();
    if (std::isfinite(det) && std::abs(det) > 1e-10f) {
        Eigen::Matrix3f inv = m_reg.inverse();
        float tr = inv.trace();
        if (std::isfinite(tr) && tr > 0.0f) {
            vdop = std::sqrt(tr);
        }
    }
    float rho = vdop / std::max(params.vdop_typ, 1e-3f);
    float w_geom_legacy =
        std::exp(-std::max(0.0f, rho - 1.0f) / std::max(params.sigma_vdop, 1e-3f));

    float w_obs = std::min(1.0f, ev.q_sum / std::max(params.n_req, 1e-3f));
    float w_v = std::clamp(w_geom * w_obs, params.w_min, 1.0f);
    return {w_v, w_geom, aniso, vdop, w_geom_legacy, w_obs};
}

// 진단 전용: 판정에 영향을 주지 않는 관측 누적기
struct EvidenceDiag {
    std::array<int, 20> wgeom_hist{};         // 0.05 폭, 신규 aniso 기반 w_geom
    std::array<int, 20> wgeom_legacy_hist{};  // 0.05 폭, 구 vDOP 기반 w_geom
    std::array<int, 20> aniso_hist{};         // 0.05 폭, lambda_2/lambda_1
    std::array<int, 11> qsum_hist{};          // 아래 kQSumEdges 경계
    std::array<int, 10> lratio_hist{};        // log10(lambda_min/lambda_max) 십진 구간
    std::vector<float> vdops;
    std::size_t w_obs_partial = 0;  // q_sum < n_req 인 voxel 수
    std::size_t vdop_invalid = 0;   // determinant 조건 실패로 vdop 미산출
};

constexpr std::array<float, 10> kQSumEdges{0.5f, 1.f, 2.f, 3.f, 5.f, 10.f, 20.f, 40.f, 80.f, 160.f};

void accumulateEvidenceDiag(const lt_mapping::VoidEvidence& ev, const WeightComponents& comp,
                            EvidenceDiag& diag)
{
    int wg_bin = std::clamp(static_cast<int>(comp.w_geom / 0.05f), 0, 19);
    diag.wgeom_hist[static_cast<size_t>(wg_bin)]++;

    int wgl_bin = std::clamp(static_cast<int>(comp.w_geom_legacy / 0.05f), 0, 19);
    diag.wgeom_legacy_hist[static_cast<size_t>(wgl_bin)]++;

    int an_bin = std::clamp(static_cast<int>(comp.aniso / 0.05f), 0, 19);
    diag.aniso_hist[static_cast<size_t>(an_bin)]++;

    size_t q_bin = 0;
    while (q_bin < kQSumEdges.size() && ev.q_sum >= kQSumEdges[q_bin]) {
        q_bin++;
    }
    diag.qsum_hist[q_bin]++;
    if (comp.w_obs < 1.0f) {
        diag.w_obs_partial++;
    }

    if (comp.vdop > 0.0f) {
        diag.vdops.push_back(comp.vdop);
    } else {
        diag.vdop_invalid++;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
    es.computeDirect(ev.M, Eigen::EigenvaluesOnly);
    const auto& lam = es.eigenvalues();
    float l_max = std::max(lam(2), 1e-30f);
    float l_min = std::max(lam(0), 0.0f);
    float ratio = l_min / l_max;
    int r_bin = (ratio <= 0.0f) ? 0
                                : std::clamp(static_cast<int>(std::floor(std::log10(ratio))) + 10, 0, 9);
    diag.lratio_hist[static_cast<size_t>(r_bin)]++;
}

void reportEvidenceDiag(EvidenceDiag& diag)
{
    auto logger = rclcpp::get_logger("long_term_mapping");

    auto binned = [](const char* label, const std::array<int, 20>& h) {
        std::ostringstream os;
        os << label;
        for (size_t i = 0; i < h.size(); ++i) {
            os << " [" << static_cast<float>(i) * 0.05f << "," << static_cast<float>(i + 1) * 0.05f
               << ")=" << h[i];
        }
        return os.str();
    };

    RCLCPP_INFO(logger, "%s", binned("w_geom histogram (aniso)", diag.wgeom_hist).c_str());
    RCLCPP_INFO(logger, "%s",
                binned("w_geom histogram (legacy vdop)", diag.wgeom_legacy_hist).c_str());
    RCLCPP_INFO(logger, "%s",
                binned("aniso histogram (lambda_2/lambda_1)", diag.aniso_hist).c_str());

    std::ostringstream qs;
    qs << "q_sum histogram [0," << kQSumEdges[0] << ")=" << diag.qsum_hist[0];
    for (size_t i = 1; i < kQSumEdges.size(); ++i) {
        qs << " [" << kQSumEdges[i - 1] << "," << kQSumEdges[i] << ")=" << diag.qsum_hist[i];
    }
    qs << " [" << kQSumEdges.back() << ",inf)=" << diag.qsum_hist[kQSumEdges.size()];
    qs << " | q_sum<n_req voxels=" << diag.w_obs_partial;
    RCLCPP_INFO(logger, "%s", qs.str().c_str());

    std::ostringstream lr;
    lr << "M eigen ratio histogram (lambda_min/lambda_max) <=1e-9=" << diag.lratio_hist[0];
    for (size_t i = 1; i < diag.lratio_hist.size(); ++i) {
        lr << " [1e-" << (10 - i) << ",1e-" << (9 - i) << ")=" << diag.lratio_hist[i];
    }
    RCLCPP_INFO(logger, "%s", lr.str().c_str());

    if (!diag.vdops.empty()) {
        std::sort(diag.vdops.begin(), diag.vdops.end());
        auto pct = [&](double p) {
            size_t i = static_cast<size_t>(p * static_cast<double>(diag.vdops.size() - 1));
            return diag.vdops[i];
        };
        RCLCPP_INFO(logger,
                    "vDOP percentiles (legacy, unused in decision): min=%.4f p10=%.4f p25=%.4f "
                    "p50=%.4f p75=%.4f p90=%.4f "
                    "max=%.4f (n=%zu, invalid=%zu)",
                    diag.vdops.front(), pct(0.10), pct(0.25), pct(0.50), pct(0.75), pct(0.90),
                    diag.vdops.back(), diag.vdops.size(), diag.vdop_invalid);
    }
}

inline float sigmoid(float x)
{
    if (x >= 0.0f) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = std::exp(x);
    return z / (1.0f + z);
}

void computeMapStateStats(const UfoVoidMap& map, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                          float res, int& void_cnt, int& hit_cnt)
{
    (void)res;
    std::unordered_set<int64_t> seen_void;
    std::unordered_set<int64_t> seen_hit;
    seen_void.reserve(cloud->points.size() / 4 + 1);
    seen_hit.reserve(cloud->points.size() / 4 + 1);

    for (const auto& p : cloud->points) {
        ufo::Point q(p.x, p.y, p.z);
        int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, res);
        if (map.seenFree(q)) {
            seen_void.insert(key);
        }
        if (map.hits(q) > 0) {
            seen_hit.insert(key);
        }
    }
    void_cnt = static_cast<int>(seen_void.size());
    hit_cnt = static_cast<int>(seen_hit.size());
}

ClassificationResult classifyChanges(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_map, const UfoVoidMap& map1,
    const UfoVoidMap& map2,
    const std::unordered_map<int64_t, lt_mapping::VoidEvidence>& ev_1in2,
    const std::unordered_map<int64_t, lt_mapping::VoidEvidence>& ev_2in1,
    const lt_mapping::EvidenceParams& params)
{
    ClassificationResult result;

    std::unordered_map<int64_t, float> w_1in2;
    std::unordered_map<int64_t, float> w_2in1;
    w_1in2.reserve(ev_1in2.size());
    w_2in1.reserve(ev_2in1.size());

    int single_kf_cnt = 0;
    double single_kf_wgeom_sum = 0.0;
    EvidenceDiag diag;
    diag.vdops.reserve(ev_1in2.size() + ev_2in1.size());

    // q_sum <= 0 인 엔트리는 DDA ray가 한 번도 통과하지 않은 voxel(프리패스로 생성된 빈 엔트리)이다.
    // computeVoidWeight()가 w_min을 반환하므로 조회 폴백(w_min)과 결과가 같다. 진단 지표의 원래
    // 의미("DDA 증거가 있는 voxel")를 유지하기 위해 가중치 맵과 진단 집계에서 제외한다.
    for (const auto& kv : ev_1in2) {
        if (kv.second.q_sum <= 0.0f) {
            continue;
        }
        auto comp = computeVoidWeight(kv.second, params);
        w_1in2.emplace(kv.first, comp.w_v);
        accumulateEvidenceDiag(kv.second, comp, diag);
        if (kv.second.n_keyframes == 1) {
            single_kf_cnt++;
            single_kf_wgeom_sum += comp.w_geom;
        }
    }
    for (const auto& kv : ev_2in1) {
        if (kv.second.q_sum <= 0.0f) {
            continue;
        }
        auto comp = computeVoidWeight(kv.second, params);
        w_2in1.emplace(kv.first, comp.w_v);
        accumulateEvidenceDiag(kv.second, comp, diag);
        if (kv.second.n_keyframes == 1) {
            single_kf_cnt++;
            single_kf_wgeom_sum += comp.w_geom;
        }
    }

    reportEvidenceDiag(diag);

    if (single_kf_cnt > 0) {
        RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                    "Single-keyframe void voxels: %d, mean w_geom=%.4f", single_kf_cnt,
                    single_kf_wgeom_sum / static_cast<double>(single_kf_cnt));
    }

    std::unordered_map<int64_t, pcl::PointXYZI> first_rep;
    std::unordered_map<int64_t, pcl::PointXYZI> second_rep;
    first_rep.reserve(first_map->size() / 4 + 1);
    second_rep.reserve(second_map->size() / 4 + 1);

    for (const auto& p : first_map->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, params.res);
        if (first_rep.find(key) == first_rep.end()) {
            first_rep.emplace(key, p);
        }
    }
    for (const auto& p : second_map->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, params.res);
        if (second_rep.find(key) == second_rep.end()) {
            second_rep.emplace(key, p);
        }
    }

    std::unordered_map<int64_t, lt_mapping::Persistence> pers_1;
    std::unordered_map<int64_t, lt_mapping::Persistence> pers_2;
    std::unordered_map<int64_t, float> p1_by_voxel;
    std::unordered_map<int64_t, float> l1_by_voxel;
    std::unordered_map<int64_t, float> l2_by_voxel;
    std::unordered_set<int64_t> void2_voxels;
    std::unordered_set<int64_t> void1_voxels;

    pers_1.reserve(first_rep.size());
    pers_2.reserve(second_rep.size());
    p1_by_voxel.reserve(first_rep.size());
    l1_by_voxel.reserve(first_rep.size());
    l2_by_voxel.reserve(second_rep.size());

    // UFOMap 의 hits() 원값에 대한 안전 상한. 실제 증거량 제한은 effectiveCount(n_sat) 가 맡고,
    // 이 cap 은 지수 함수 입력이 비정상적으로 커지는 것을 막는 방어선으로만 남는다.
    constexpr int kHitCountCap = 10;
    auto voidEvidenceCount = [](const std::unordered_map<int64_t, lt_mapping::VoidEvidence>& ev_map,
                                int64_t key) -> float {
        auto it = ev_map.find(key);
        if (it == ev_map.end()) {
            return 1.0f;
        }
        return std::max(1.0f, static_cast<float>(it->second.n_keyframes));
    };

    // sigmoid(l) < tau  <=>  l < logit(tau). 판정 여유(margin)를 log-odds 축에서 재기 위한 임계.
    auto logit = [](float t) {
        float c = std::clamp(t, 1e-6f, 1.0f - 1e-6f);
        return std::log(c / (1.0f - c));
    };
    const float l_tau_del = logit(params.tau_del);  // 음수
    const float l_tau_add = logit(params.tau_add);  // 양수

    // [진단] void 증거 계수의 대칭성. pers_1 은 -w_v*l_void*eff, pers_2 는 +w_v*l_void*eff 를 쓰므로
    // 순수 void voxel(n_hit=0, eff->n_sat) 기준 유효 문턱은 양쪽 모두 |logit(tau)|/(l_void*n_sat) 다.
    {
        const float denom = std::max(params.l_void * params.n_sat, 1e-6f);
        const float nd_wv_thr = -l_tau_del / denom;
        const float pd_wv_thr = l_tau_add / denom;
        RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                    "PD coefficient: pers2 void branch uses l_void=%.3f (center and dilate), hit "
                    "branch uses l_hit=%.3f | pure-void w_v threshold: ND>%.4f PD>%.4f "
                    "(|logit(tau_del)|=%.4f, logit(tau_add)=%.4f, n_sat=%.2f) -> symmetric=%s",
                    params.l_void, params.l_hit, nd_wv_thr, pd_wv_thr, -l_tau_del, l_tau_add,
                    params.n_sat,
                    std::abs(nd_wv_thr - pd_wv_thr) < 1e-4f ? "yes" : "NO");
    }

    std::size_t void_total_1 = 0;
    std::size_t void_fallback_1 = 0;
    std::size_t void_total_2 = 0;
    std::size_t void_fallback_2 = 0;

    // 진단 전용 카운터: clamp 발생 분기 분리 및 clamp가 판정을 뒤집었는지 여부
    std::size_t hit_branch_1 = 0, hit_branch_2 = 0;
    std::size_t clamp_hit_1 = 0, clamp_void_1 = 0, clamp_flip_1 = 0;
    std::size_t clamp_hit_2 = 0, clamp_void_2 = 0, clamp_flip_2 = 0;

    // 진단 전용: 경합 구조 전환의 실제 규모와 판정 여유를 정량화
    std::size_t overlap_1 = 0, overlap_2 = 0;  // is_void && n_hit>0 인 voxel 수
    std::array<int, 20> void_wv_hist_1{};      // void 판정 voxel 만의 w_v 분포
    std::array<int, 20> void_wv_hist_2{};
    std::vector<float> nvoid_raw_1, nvoid_eff_1, nvoid_raw_2, nvoid_eff_2;
    std::vector<float> margin_1, margin_2;  // 양수면 ND/PD 성립, 크기는 임계로부터의 거리
    nvoid_raw_1.reserve(first_rep.size() / 8 + 1);
    nvoid_eff_1.reserve(first_rep.size() / 8 + 1);
    nvoid_raw_2.reserve(second_rep.size() / 8 + 1);
    nvoid_eff_2.reserve(second_rep.size() / 8 + 1);
    margin_1.reserve(first_rep.size() / 8 + 1);
    margin_2.reserve(second_rep.size() / 8 + 1);
    auto wvBin = [](float w) { return static_cast<size_t>(std::clamp(static_cast<int>(w / 0.05f), 0, 19)); };

    // [이웃 관용 교차 조회]
    // 입력 맵은 VOXEL_SIZE(0.4 m) 로 다운샘플되어 있어 두 세션이 같은 물리 표면을 봐도 centroid 가
    // 판정 격자(res=0.2 m) 기준 한 칸 어긋날 수 있고, 맵 삽입의 inflate_unknown=1 이 표면 주위에
    // 1 voxel 두께의 unknown 껍질을 만든다. 두 효과의 스케일이 겹치므로 "중심에서 미관측"의 상당
    // 부분은 진짜 미관측이 아니라 정렬 오차다(실측: UE 의 40~48% 가 이웃 칸에 상대 세션 hit 보유).
    // 중심에서 아무 상태도 못 찾은 voxel 에 한해 이웃까지 조회를 넓히되, 위치 추론에 기반한
    // 증거이므로 dilate_weight 로 감쇠시킨다.
    struct DilateProbe {
        int hit = 0;              // 이웃 중 최대 hits (0 이면 hit 이웃 없음)
        bool free_found = false;  // hit 이 아닌 seenFree 이웃 존재 여부
    };
    int64_t dilate_ns = 0;
    std::size_t dilate_queries = 0;
    auto dilateProbe = [&](const pcl::PointXYZI& p, const UfoVoidMap& other_map) -> DilateProbe {
        const auto t0 = std::chrono::steady_clock::now();
        DilateProbe out;
        const int r = params.cross_dilate;
        for (int dz = -r; dz <= r; ++dz) {
            for (int dy = -r; dy <= r; ++dy) {
                for (int dx = -r; dx <= r; ++dx) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    ufo::Point q(p.x + static_cast<float>(dx) * params.res,
                                 p.y + static_cast<float>(dy) * params.res,
                                 p.z + static_cast<float>(dz) * params.res);
                    // hit 우선. 정렬 오차로 표면이 옆 칸에 있는 경우가 지배적이므로 hit 을 먼저 본다.
                    const int h =
                        std::clamp(std::max(0, static_cast<int>(other_map.hits(q))), 0, kHitCountCap);
                    if (h > 0) {
                        out.hit = std::max(out.hit, h);
                    } else if (other_map.seenFree(q)) {
                        out.free_found = true;
                    }
                    ++dilate_queries;
                }
            }
        }
        dilate_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
        return out;
    };

    // 이웃 승계 진단 카운터. center_miss 는 승계 이전(=기존 구조에서의) UE 후보 수다.
    std::size_t center_miss_1 = 0, center_miss_2 = 0;
    std::size_t dilate_hit_1 = 0, dilate_void_1 = 0;
    std::size_t dilate_hit_2 = 0, dilate_void_2 = 0;
    std::size_t dilate_nd_1 = 0, dilate_nd_from_hit_1 = 0, dilate_static_1 = 0;
    std::size_t dilate_pd_2 = 0, dilate_pd_from_hit_2 = 0, dilate_static_2 = 0;

    // pers_1 명제: "1세션 구조물이 여전히 존재한다(정적이다)"
    for (const auto& kv : first_rep) {
        int64_t key = kv.first;
        const auto& p = kv.second;
        auto& ps = pers_1[key];
        const float l_prev = ps.log_odds;

        float w_v = params.w_min;
        auto wit = w_1in2.find(key);
        bool w_missing = (wit == w_1in2.end());
        if (wit != w_1in2.end()) {
            w_v = wit->second;
        }

        ufo::Point q(p.x, p.y, p.z);
        // SEEN_FREE 와 REFLECTION 은 독립 레이어이므로 한 voxel 이 동시에 참일 수 있다.
        // 두 증거를 모두 누적해 경합시킨다(배타 분기는 우세한 쪽 증거를 통째로 버린다).
        bool is_void = map2.seenFree(q);
        int n_hit = std::clamp(std::max(0, static_cast<int>(map2.hits(q))), 0, kHitCountCap);

        float raw_l = l_prev;
        bool observed = false;

        if (n_hit > 0) {
            const float n_hit_eff =
                lt_mapping::effectiveCount(static_cast<float>(n_hit), params.n_sat);
            raw_l += params.l_hit * n_hit_eff;  // 존재 증거
            ++hit_branch_1;
            if (ps.hit_cnt < std::numeric_limits<uint16_t>::max()) {
                ps.hit_cnt = static_cast<uint16_t>(
                    std::min<int>(std::numeric_limits<uint16_t>::max(), ps.hit_cnt + n_hit));
            }
            observed = true;
        }
        if (is_void) {
            ++void_total_1;
            if (w_missing) {
                ++void_fallback_1;
            }
            const float n_void = voidEvidenceCount(ev_1in2, key);
            const float n_void_eff = lt_mapping::effectiveCount(n_void, params.n_sat);
            raw_l -= w_v * params.l_void * n_void_eff;  // 소멸 증거
            ps.void_w_sum += w_v;
            void2_voxels.insert(key);
            observed = true;

            if (n_hit > 0) {
                ++overlap_1;
            }
            void_wv_hist_1[wvBin(w_v)]++;
            nvoid_raw_1.push_back(n_void);
            nvoid_eff_1.push_back(n_void_eff);
        }

        // 중심 우선: 중심에서 hit 또는 seenFree 를 하나라도 찾았으면 이웃은 보지 않는다.
        int dilate_kind = 0;  // 0=미적용, 1=이웃 hit 승계, 2=이웃 void 승계
        if (!observed) {
            ++center_miss_1;
            if (params.cross_dilate > 0) {
                const DilateProbe nb = dilateProbe(p, map2);
                if (nb.hit > 0) {
                    const float n_hit_eff =
                        lt_mapping::effectiveCount(static_cast<float>(nb.hit), params.n_sat);
                    raw_l += params.dilate_weight * params.l_hit * n_hit_eff;
                    dilate_kind = 1;
                    ++dilate_hit_1;
                } else if (nb.free_found) {
                    // n_void 는 중심 voxel 의 ev_1in2 엔트리에서 가져온다. 함께 곱해지는 w_v 가
                    // 중심 voxel 의 DDA 증거로 계산된 값이므로, 같은 출처를 써야 정합적이다.
                    const float n_void = voidEvidenceCount(ev_1in2, key);
                    const float n_void_eff = lt_mapping::effectiveCount(n_void, params.n_sat);
                    raw_l -= params.dilate_weight * w_v * params.l_void * n_void_eff;
                    dilate_kind = 2;
                    ++dilate_void_1;
                }
                if (dilate_kind != 0) {
                    observed = true;
                }
            }
        }

        ps.log_odds = clampLogOdds(raw_l, params.l_max);
        if (std::abs(raw_l) > params.l_max) {
            if (is_void) {
                ++clamp_void_1;
            } else {
                ++clamp_hit_1;
            }
            if ((sigmoid(raw_l) < params.tau_del) != (sigmoid(ps.log_odds) < params.tau_del)) {
                ++clamp_flip_1;
            }
        }

        if (!observed) {
            result.first_ue_voxels.insert(key);
        }

        float p_v = sigmoid(ps.log_odds);
        p1_by_voxel.emplace(key, p_v);
        l1_by_voxel.emplace(key, ps.log_odds);
        if (is_void) {
            margin_1.push_back(l_tau_del - ps.log_odds);
        }
        if (p_v < params.tau_del) {
            result.nd_voxels.insert(key);
            if (dilate_kind != 0) {
                ++dilate_nd_1;
                if (dilate_kind == 1) {
                    ++dilate_nd_from_hit_1;
                }
            }
        } else if (dilate_kind != 0) {
            ++dilate_static_1;
        }
    }

    // pers_2 명제: "2세션 구조물이 새로 생긴 것이다"
    for (const auto& kv : second_rep) {
        int64_t key = kv.first;
        const auto& p = kv.second;
        auto& ps = pers_2[key];
        const float l_prev = ps.log_odds;

        float w_v = params.w_min;
        auto wit = w_2in1.find(key);
        bool w_missing = (wit == w_2in1.end());
        if (wit != w_2in1.end()) {
            w_v = wit->second;
        }

        ufo::Point q(p.x, p.y, p.z);
        bool is_void = map1.seenFree(q);
        int n_hit = std::clamp(std::max(0, static_cast<int>(map1.hits(q))), 0, kHitCountCap);

        float raw_l = l_prev;
        bool observed = false;

        if (n_hit > 0) {
            const float n_hit_eff =
                lt_mapping::effectiveCount(static_cast<float>(n_hit), params.n_sat);
            raw_l -= params.l_hit * n_hit_eff;  // 이전에도 있었다는 증거 -> 신규 아님
            ++hit_branch_2;
            if (ps.hit_cnt < std::numeric_limits<uint16_t>::max()) {
                ps.hit_cnt = static_cast<uint16_t>(
                    std::min<int>(std::numeric_limits<uint16_t>::max(), ps.hit_cnt + n_hit));
            }
            observed = true;
        }
        if (is_void) {
            ++void_total_2;
            if (w_missing) {
                ++void_fallback_2;
            }
            const float n_void = voidEvidenceCount(ev_2in1, key);
            const float n_void_eff = lt_mapping::effectiveCount(n_void, params.n_sat);
            // pers_1 의 void 분기(-w_v * l_void * eff)와 크기가 같고 부호만 반대다. void 증거는
            // 어느 방향으로 읽든 같은 종류의 증거이므로 계수도 같아야 하고, 그래야 ND 와 PD 의
            // 유효 문턱이 |logit(tau)| / (l_void * n_sat) 로 일치한다.
            raw_l += w_v * params.l_void * n_void_eff;  // 이전에 비어 있었다는 증거 -> 신규
            ps.void_w_sum += w_v;
            void1_voxels.insert(key);
            observed = true;

            if (n_hit > 0) {
                ++overlap_2;
            }
            void_wv_hist_2[wvBin(w_v)]++;
            nvoid_raw_2.push_back(n_void);
            nvoid_eff_2.push_back(n_void_eff);
        }

        // pers_1 과 부호 대칭. 이웃 hit 은 "이전에도 있었다" -> 신규 아님(음), 이웃 free 는
        // "이전에 비어 있었다" -> 신규(양).
        int dilate_kind = 0;
        if (!observed) {
            ++center_miss_2;
            if (params.cross_dilate > 0) {
                const DilateProbe nb = dilateProbe(p, map1);
                if (nb.hit > 0) {
                    const float n_hit_eff =
                        lt_mapping::effectiveCount(static_cast<float>(nb.hit), params.n_sat);
                    raw_l -= params.dilate_weight * params.l_hit * n_hit_eff;
                    dilate_kind = 1;
                    ++dilate_hit_2;
                } else if (nb.free_found) {
                    const float n_void = voidEvidenceCount(ev_2in1, key);
                    const float n_void_eff = lt_mapping::effectiveCount(n_void, params.n_sat);
                    raw_l += params.dilate_weight * w_v * params.l_void * n_void_eff;
                    dilate_kind = 2;
                    ++dilate_void_2;
                }
                if (dilate_kind != 0) {
                    observed = true;
                }
            }
        }

        ps.log_odds = clampLogOdds(raw_l, params.l_max);
        if (std::abs(raw_l) > params.l_max) {
            if (is_void) {
                ++clamp_void_2;
            } else {
                ++clamp_hit_2;
            }
            if ((sigmoid(raw_l) > params.tau_add) != (sigmoid(ps.log_odds) > params.tau_add)) {
                ++clamp_flip_2;
            }
        }

        if (!observed) {
            result.second_ue_voxels.insert(key);
        }

        float p_v = sigmoid(ps.log_odds);
        l2_by_voxel.emplace(key, ps.log_odds);
        if (is_void) {
            margin_2.push_back(ps.log_odds - l_tau_add);
        }
        // is_void 게이트를 제거해 ND 조건(p_v < tau_del)과 대칭을 맞춘다. log_odds 는 0 에서
        // 출발하고 양의 기여는 여전히 void 증거뿐이다(중심 seenFree 또는 이웃 승계 free). 이웃
        // 승계는 배타적이라 hit 승계와 동시에 일어나지 않으므로, p_v > tau_add 는 "1세션이 이
        // 자리(또는 그 이웃)를 비어 있다고 관측했다"를 여전히 함의한다. 게이트는 계속 중복이다.
        // void 계수를 l_hit 에서 l_void 로 바꾼 뒤에도 두 양의 항의 부호는 그대로이고(l_void>0,
        // w_v>=w_min>0, eff>0) 음의 항은 hit 뿐이므로 이 증명은 유지된다.
        if (p_v > params.tau_add) {
            result.pd_voxels.insert(key);
            if (dilate_kind != 0) {
                ++dilate_pd_2;
                if (dilate_kind == 1) {
                    ++dilate_pd_from_hit_2;
                }
            }
        } else if (dilate_kind != 0) {
            ++dilate_static_2;
        }
    }

    for (const auto& p : first_map->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, params.res);

        if (result.first_ue_voxels.find(key) != result.first_ue_voxels.end()) {
            result.first_ue->points.push_back(p);
        }
        if (result.nd_voxels.find(key) != result.nd_voxels.end()) {
            result.nd_cloud->points.push_back(p);
        }

        auto pv_it = p1_by_voxel.find(key);
        if (pv_it != p1_by_voxel.end()) {
            pcl::PointXYZI p_dbg = p;
            p_dbg.intensity = pv_it->second * 100.0f;
            result.persistence_cloud->points.push_back(p_dbg);
        }

        if (void2_voxels.find(key) != void2_voxels.end()) {
            auto w_it = w_1in2.find(key);
            float w_v = (w_it != w_1in2.end()) ? w_it->second : params.w_min;
            pcl::PointXYZI dbg = p;
            dbg.intensity = w_v * 100.0f;
            result.evidence_debug_cloud->points.push_back(dbg);
        }
    }

    for (const auto& p : second_map->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, params.res);

        if (result.second_ue_voxels.find(key) != result.second_ue_voxels.end()) {
            result.second_ue->points.push_back(p);
        }
        if (result.pd_voxels.find(key) != result.pd_voxels.end()) {
            result.pd_cloud->points.push_back(p);
        }
    }

    std::array<int, 20> hist{};
    for (const auto& kv : w_1in2) {
        int bin = std::clamp(static_cast<int>(kv.second / 0.05f), 0, 19);
        hist[static_cast<size_t>(bin)]++;
    }
    for (const auto& kv : w_2in1) {
        int bin = std::clamp(static_cast<int>(kv.second / 0.05f), 0, 19);
        hist[static_cast<size_t>(bin)]++;
    }

    std::ostringstream oss;
    oss << "w_v histogram";
    for (size_t i = 0; i < hist.size(); ++i) {
        float b0 = static_cast<float>(i) * 0.05f;
        float b1 = b0 + 0.05f;
        oss << " [" << b0 << "," << b1 << ")=" << hist[i];
    }
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"), "%s", oss.str().c_str());

    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Void evidence fallback (1-in-2): %zu / %zu voxels (%.2f%%) had no DDA evidence -> w_min",
                void_fallback_1, void_total_1,
                void_total_1 > 0 ? 100.0 * static_cast<double>(void_fallback_1) /
                                       static_cast<double>(void_total_1)
                                 : 0.0);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Void evidence fallback (2-in-1): %zu / %zu voxels (%.2f%%) had no DDA evidence -> w_min",
                void_fallback_2, void_total_2,
                void_total_2 > 0 ? 100.0 * static_cast<double>(void_fallback_2) /
                                       static_cast<double>(void_total_2)
                                 : 0.0);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Evidence coverage: w_1in2=%zu voxels, w_2in1=%zu voxels",
                w_1in2.size(), w_2in1.size());

    auto keyframeStat = [](const std::unordered_set<int64_t>& keys,
                           const std::unordered_map<int64_t, lt_mapping::VoidEvidence>& ev_map)
        -> std::tuple<int, double, int> {
        if (keys.empty()) {
            return std::make_tuple(0, 0.0, 0);
        }
        int min_v = std::numeric_limits<int>::max();
        int max_v = 0;
        double sum_v = 0.0;
        for (int64_t key : keys) {
            int v = 1;
            auto it = ev_map.find(key);
            if (it != ev_map.end()) {
                v = std::max(1, static_cast<int>(it->second.n_keyframes));
            }
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
            sum_v += static_cast<double>(v);
        }
        return std::make_tuple(min_v, sum_v / static_cast<double>(keys.size()), max_v);
    };

    auto [nd_kf_min, nd_kf_avg, nd_kf_max] = keyframeStat(result.nd_voxels, ev_1in2);
    auto [pd_kf_min, pd_kf_avg, pd_kf_max] = keyframeStat(result.pd_voxels, ev_2in1);

    std::size_t neg_tail = 0;
    std::size_t pos_tail = 0;
    std::size_t clamp_hit = 0;
    for (const auto& kv : l1_by_voxel) {
        if (kv.second <= -0.8f) {
            neg_tail++;
        }
        if (std::abs(kv.second) >= params.l_max - 1e-4f) {
            clamp_hit++;
        }
    }
    for (const auto& kv : l2_by_voxel) {
        if (kv.second >= 0.8f) {
            pos_tail++;
        }
        if (std::abs(kv.second) >= params.l_max - 1e-4f) {
            clamp_hit++;
        }
    }

    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Evidence count: raw n_hit cap=%d, n_void from n_keyframes, "
                "both via effectiveCount(n_sat=%.2f)",
                kHitCountCap, params.n_sat);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "ND n_keyframes(min/avg/max)=%d/%.2f/%d, PD n_keyframes(min/avg/max)=%d/%.2f/%d",
                nd_kf_min, nd_kf_avg, nd_kf_max, pd_kf_min, pd_kf_avg, pd_kf_max);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Log-odds tails: neg<=-0.8=%zu, pos>=0.8=%zu, clamp_hits=%zu",
                neg_tail, pos_tail, clamp_hit);
    // 경합 구조에서는 hit_branch 와 void_branch 가 더 이상 배타적이지 않다(overlap 만큼 겹친다).
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Clamp split: pers1 hit_branch=%zu(clamped=%zu) void_branch=%zu(clamped=%zu) "
                "flips=%zu | pers2 hit_branch=%zu(clamped=%zu) void_branch=%zu(clamped=%zu) flips=%zu",
                hit_branch_1, clamp_hit_1, void_total_1, clamp_void_1, clamp_flip_1, hit_branch_2,
                clamp_hit_2, void_total_2, clamp_void_2, clamp_flip_2);

    // [진단 1] 배타 분기가 버리던 증거의 실제 규모. 값이 크면 이전 구조가 대량 오판을 냈다는 뜻.
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Void-hit overlap: pers1 %zu voxels (%.2f%% of void, %.2f%% of hit) | "
                "pers2 %zu voxels (%.2f%% of void, %.2f%% of hit)",
                overlap_1,
                void_total_1 > 0 ? 100.0 * static_cast<double>(overlap_1) /
                                       static_cast<double>(void_total_1)
                                 : 0.0,
                hit_branch_1 > 0 ? 100.0 * static_cast<double>(overlap_1) /
                                       static_cast<double>(hit_branch_1)
                                 : 0.0,
                overlap_2,
                void_total_2 > 0 ? 100.0 * static_cast<double>(overlap_2) /
                                       static_cast<double>(void_total_2)
                                 : 0.0,
                hit_branch_2 > 0 ? 100.0 * static_cast<double>(overlap_2) /
                                       static_cast<double>(hit_branch_2)
                                 : 0.0);

    // [진단 2] 판정에 실제로 쓰이는 분포. 전체 관심 voxel 대상인 "w_v histogram" 과 분리한다.
    auto wvHistLine = [](const char* label, const std::array<int, 20>& h) {
        std::ostringstream os;
        os << label;
        for (size_t i = 0; i < h.size(); ++i) {
            os << " [" << static_cast<float>(i) * 0.05f << "," << static_cast<float>(i + 1) * 0.05f
               << ")=" << h[i];
        }
        return os.str();
    };
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"), "%s",
                wvHistLine("Void voxel w_v distribution (pers1)", void_wv_hist_1).c_str());
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"), "%s",
                wvHistLine("Void voxel w_v distribution (pers2)", void_wv_hist_2).c_str());

    // [진단 3] 상관 관측 보정 전/후 대조. eff 가 n_sat 에 붙어 있으면 보정이 포화 구간에서 동작 중.
    auto pctStat = [](std::vector<float>& v) -> std::array<float, 5> {
        if (v.empty()) {
            return {0.f, 0.f, 0.f, 0.f, 0.f};
        }
        std::sort(v.begin(), v.end());
        auto at = [&](double f) {
            return v[static_cast<size_t>(f * static_cast<double>(v.size() - 1))];
        };
        return {v.front(), at(0.25), at(0.50), at(0.75), v.back()};
    };
    auto nr1 = pctStat(nvoid_raw_1);
    auto ne1 = pctStat(nvoid_eff_1);
    auto nr2 = pctStat(nvoid_raw_2);
    auto ne2 = pctStat(nvoid_eff_2);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "n_void effective (pers1, n=%zu): raw min/p25/p50/p75/max=%.1f/%.1f/%.1f/%.1f/%.1f "
                "-> eff=%.2f/%.2f/%.2f/%.2f/%.2f",
                nvoid_raw_1.size(), nr1[0], nr1[1], nr1[2], nr1[3], nr1[4], ne1[0], ne1[1], ne1[2],
                ne1[3], ne1[4]);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "n_void effective (pers2, n=%zu): raw min/p25/p50/p75/max=%.1f/%.1f/%.1f/%.1f/%.1f "
                "-> eff=%.2f/%.2f/%.2f/%.2f/%.2f",
                nvoid_raw_2.size(), nr2[0], nr2[1], nr2[2], nr2[3], nr2[4], ne2[0], ne2[1], ne2[2],
                ne2[3], ne2[4]);

    // [진단 4] void 후보의 판정 여유. 양수면 ND/PD 성립이고, 0 근처가 많으면 임계 튜닝 여지가 크다.
    auto marginLine = [&](const char* label, std::vector<float>& m) {
        std::size_t positive = 0, near_thr = 0;
        for (float v : m) {
            if (v > 0.0f) {
                ++positive;
            }
            if (std::abs(v) < 0.2f) {
                ++near_thr;
            }
        }
        auto s = pctStat(m);
        std::ostringstream os;
        os << label << ": n=" << m.size() << " min/p25/p50/p75/max=" << s[0] << "/" << s[1] << "/"
           << s[2] << "/" << s[3] << "/" << s[4] << " positive=" << positive
           << " near_threshold(|m|<0.2)=" << near_thr;
        return os.str();
    };
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"), "%s",
                marginLine("Decision margin (pers1 void, +=ND)", margin_1).c_str());
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"), "%s",
                marginLine("Decision margin (pers2 void, +=PD)", margin_2).c_str());

    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Voxel classification: ND=%zu, PD=%zu, UE1=%zu, UE2=%zu, void2=%zu, void1=%zu",
                result.nd_voxels.size(), result.pd_voxels.size(), result.first_ue_voxels.size(),
                result.second_ue_voxels.size(), void2_voxels.size(), void1_voxels.size());

    // [진단 5] void 판정이 그대로 ND/PD 로 흘러가는지(=가중치가 아무것도 거르지 못하는지) 직접 표시.
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "ND/PD yield: ND/void2=%zu/%zu (%.2f%%), PD/void1=%zu/%zu (%.2f%%)",
                result.nd_voxels.size(), void2_voxels.size(),
                void2_voxels.empty() ? 0.0
                                     : 100.0 * static_cast<double>(result.nd_voxels.size()) /
                                           static_cast<double>(void2_voxels.size()),
                result.pd_voxels.size(), void1_voxels.size(),
                void1_voxels.empty() ? 0.0
                                     : 100.0 * static_cast<double>(result.pd_voxels.size()) /
                                           static_cast<double>(void1_voxels.size()));

    // [진단 6] 이웃 관용 조회가 실제로 해소한 UE 규모와 그 성격
    const std::size_t dilate_res_1 = dilate_hit_1 + dilate_void_1;
    const std::size_t dilate_res_2 = dilate_hit_2 + dilate_void_2;
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Dilate resolution: pers1 candidates=%zu resolved=%zu (hit=%zu void=%zu) "
                "remain_UE=%zu | pers2 candidates=%zu resolved=%zu (hit=%zu void=%zu) remain_UE=%zu "
                "(cross_dilate=%d, dilate_weight=%.2f)",
                center_miss_1, dilate_res_1, dilate_hit_1, dilate_void_1,
                result.first_ue_voxels.size(), center_miss_2, dilate_res_2, dilate_hit_2,
                dilate_void_2, result.second_ue_voxels.size(), params.cross_dilate,
                params.dilate_weight);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "UE reduction: UE1 %zu -> %zu (-%zu, -%.2f%%) | UE2 %zu -> %zu (-%zu, -%.2f%%)",
                center_miss_1, result.first_ue_voxels.size(), dilate_res_1,
                center_miss_1 > 0
                    ? 100.0 * static_cast<double>(dilate_res_1) / static_cast<double>(center_miss_1)
                    : 0.0,
                center_miss_2, result.second_ue_voxels.size(), dilate_res_2,
                center_miss_2 > 0
                    ? 100.0 * static_cast<double>(dilate_res_2) / static_cast<double>(center_miss_2)
                    : 0.0);
    // hit 승계는 pers1 에서 항상 양(존재 증거)이므로 ND 로 갈 수 없고, pers2 에서 항상 음이므로
    // PD 로 갈 수 없다. from_hit 이 0 이 아니면 부호 규약이 깨진 것이다.
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Dilated evidence outcome: pers1 ND=%zu(from_hit=%zu) static=%zu | "
                "pers2 PD=%zu(from_hit=%zu) static=%zu",
                dilate_nd_1, dilate_nd_from_hit_1, dilate_static_1, dilate_pd_2,
                dilate_pd_from_hit_2, dilate_static_2);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Dilate elapsed: %.3f sec (%zu probes, %zu neighbor queries)",
                static_cast<double>(dilate_ns) / 1e9,
                params.cross_dilate > 0 ? center_miss_1 + center_miss_2 : 0u, dilate_queries);

    // UE voxel 이웃 상태 진단: 상대 세션이 정말 미관측인지, 아니면 1 voxel 어긋난 표면인지 구분
    auto ueNeighborStat = [&](const std::unordered_set<int64_t>& ue_keys,
                              const std::unordered_map<int64_t, pcl::PointXYZI>& rep,
                              const UfoVoidMap& other_map) {
        std::size_t adj_hit = 0;
        std::size_t adj_free_only = 0;
        std::size_t adj_none = 0;
        for (int64_t key : ue_keys) {
            auto it = rep.find(key);
            if (it == rep.end()) {
                continue;
            }
            const auto& p = it->second;
            bool hit_found = false;
            bool free_found = false;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0 && dz == 0) {
                            continue;
                        }
                        ufo::Point q(p.x + static_cast<float>(dx) * params.res,
                                     p.y + static_cast<float>(dy) * params.res,
                                     p.z + static_cast<float>(dz) * params.res);
                        if (other_map.hits(q) > 0) {
                            hit_found = true;
                        } else if (other_map.seenFree(q)) {
                            free_found = true;
                        }
                    }
                }
            }
            if (hit_found) {
                adj_hit++;
            } else if (free_found) {
                adj_free_only++;
            } else {
                adj_none++;
            }
        }
        return std::make_tuple(adj_hit, adj_free_only, adj_none);
    };

    const auto ue_t0 = std::chrono::steady_clock::now();
    auto [ue1_adj_hit, ue1_adj_free, ue1_adj_none] =
        ueNeighborStat(result.first_ue_voxels, first_rep, map2);
    auto [ue2_adj_hit, ue2_adj_free, ue2_adj_none] =
        ueNeighborStat(result.second_ue_voxels, second_rep, map1);
    const auto ue_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - ue_t0)
                           .count();
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "UE 26-neighbor state (after dilate): UE1 adj_hit=%zu adj_free_only=%zu "
                "adj_none=%zu | UE2 adj_hit=%zu adj_free_only=%zu adj_none=%zu (%.2f sec)",
                ue1_adj_hit, ue1_adj_free, ue1_adj_none, ue2_adj_hit, ue2_adj_free, ue2_adj_none,
                static_cast<double>(ue_ms) / 1000.0);

    result.void2_total = void2_voxels.size();
    result.void1_total = void1_voxels.size();

    return result;
}

// ─── ND/PD 연결 성분 후처리 ────────────────────────────────────────────────────
// 군집 크기 히스토그램 구간: 1, 2, 3-5, 6-10, 11-30, 31-100, 101-300, 301-1000, 1000+
constexpr std::size_t kClusterBinCount = 9;

inline std::size_t clusterSizeBin(std::size_t n)
{
    if (n <= 1) return 0;
    if (n == 2) return 1;
    if (n <= 5) return 2;
    if (n <= 10) return 3;
    if (n <= 30) return 4;
    if (n <= 100) return 5;
    if (n <= 300) return 6;
    if (n <= 1000) return 7;
    return 8;
}

inline const char* clusterBinLabel(std::size_t i)
{
    static const char* const kLabels[kClusterBinCount] = {
        "1", "2", "3-5", "6-10", "11-30", "31-100", "101-300", "301-1000", "1000+"};
    return kLabels[std::min(i, kClusterBinCount - 1)];
}

struct ClusterFilterStats {
    std::size_t points_before = 0;
    std::size_t points_after = 0;
    std::size_t voxels_before = 0;
    std::size_t voxels_after = 0;
    std::size_t points_removed = 0;
    std::size_t clusters_total = 0;
    std::size_t clusters_removed = 0;
    std::array<std::size_t, kClusterBinCount> cluster_hist{};  // 필터 전 군집 수 분포
    std::array<std::size_t, kClusterBinCount> point_hist{};    // 필터 전 점 수 분포
    double elapsed_sec = 0.0;
    bool applied = false;
};

// ND/PD voxel 집합에서 연결 성분 크기가 min_cluster 미만인 군집을 제거한다.
// 정합 오차로 생긴 껍질형 오검출은 상대 세션 표면에 붙은 낱개~소형 군집으로 나타나므로
// 크기 기준만으로 효과적으로 걸러진다.
//
// voxels 와 cloud 를 함께 갱신한다. classifyChanges() 는 "cloud == 키가 voxels 에 있는 모든 점"
// 이라는 불변식으로 두 자료구조를 만들었고, composeFinalMap() 은 voxels 만, 디버그 pcd 는 cloud 만
// 참조하므로 둘이 어긋나면 최종 지도와 리포트가 서로 다른 판정을 담게 된다. 그래서 생존 판정은
// 점이 아니라 voxel 단위로 확정하고(한 voxel 의 점 중 하나라도 큰 군집에 속하면 그 voxel 생존),
// cloud 는 생존 voxel 의 점을 전부 남기는 방식으로 불변식을 그대로 유지한다.
//
// min_cluster <= 1 이면 군집 크기 통계만 계산하고 voxels/cloud 는 한 바이트도 건드리지 않는다.
ClusterFilterStats filterSmallClusters(std::unordered_set<int64_t>& voxels,
                                       pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, float eps,
                                       int min_cluster, float res)
{
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&t0]() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::steady_clock::now() - t0)
                   .count() /
               1e6;
    };

    ClusterFilterStats st;
    st.voxels_before = voxels.size();
    st.points_before = cloud ? cloud->points.size() : 0;
    st.voxels_after = st.voxels_before;
    st.points_after = st.points_before;

    // 빈 입력을 KD-tree 에 넣으면 죽는다. eps <= 0 이면 이웃 관계 자체가 정의되지 않는다.
    if (!cloud || cloud->points.empty() || eps <= 0.0f) {
        st.elapsed_sec = elapsed();
        return st;
    }

    const std::size_t n = cloud->points.size();

    // union-find 로 반경 eps 이내 연결 성분을 만든다.
    std::vector<int> parent(n);
    for (std::size_t i = 0; i < n; ++i) {
        parent[i] = static_cast<int>(i);
    }
    auto find = [&parent](int x) {
        int root = x;
        while (parent[root] != root) {
            root = parent[root];
        }
        while (parent[x] != root) {
            const int next = parent[x];
            parent[x] = root;
            x = next;
        }
        return root;
    };

    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(cloud);
    pcl::Indices nn_idx;
    std::vector<float> nn_d2;
    for (std::size_t i = 0; i < n; ++i) {
        nn_idx.clear();
        nn_d2.clear();
        if (tree.radiusSearch(cloud->points[i], static_cast<double>(eps), nn_idx, nn_d2) <= 0) {
            continue;
        }
        // ri 는 이 루프 동안 계속 root 다(다른 root 를 ri 밑에 붙이기만 하므로).
        const int ri = find(static_cast<int>(i));
        for (const auto& j : nn_idx) {
            const int rj = find(static_cast<int>(j));
            if (rj != ri) {
                parent[rj] = ri;
            }
        }
    }

    std::unordered_map<int, std::size_t> comp_size;
    comp_size.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        comp_size[find(static_cast<int>(i))]++;
    }
    st.clusters_total = comp_size.size();
    for (const auto& kv : comp_size) {
        const std::size_t b = clusterSizeBin(kv.second);
        st.cluster_hist[b]++;
        st.point_hist[b] += kv.second;
    }

    // A/B 기준선: 필터 비활성. 위에서 계산한 것은 읽기 전용 통계뿐이다.
    if (min_cluster <= 1) {
        st.elapsed_sec = elapsed();
        return st;
    }
    st.applied = true;

    const std::size_t min_size = static_cast<std::size_t>(min_cluster);
    for (const auto& kv : comp_size) {
        if (kv.second < min_size) {
            st.clusters_removed++;
        }
    }

    std::unordered_set<int64_t> keep_voxels;
    keep_voxels.reserve(voxels.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (comp_size[find(static_cast<int>(i))] < min_size) {
            continue;
        }
        const auto& p = cloud->points[i];
        const int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, res);
        // cloud 에만 있고 voxels 에는 없는 점이 있더라도 voxel 집합을 늘리지 않는다.
        if (voxels.find(key) != voxels.end()) {
            keep_voxels.insert(key);
        }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr kept(new pcl::PointCloud<pcl::PointXYZI>());
    kept->points.reserve(n);
    for (const auto& p : cloud->points) {
        const int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, res);
        if (keep_voxels.find(key) != keep_voxels.end()) {
            kept->points.push_back(p);
        }
    }
    kept->width = static_cast<uint32_t>(kept->points.size());
    kept->height = 1;
    kept->is_dense = false;

    cloud = kept;
    voxels.swap(keep_voxels);

    st.points_after = cloud->points.size();
    st.voxels_after = voxels.size();
    st.points_removed = st.points_before - st.points_after;
    st.elapsed_sec = elapsed();
    return st;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr composeFinalMap(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_map, const ClassificationResult& cls,
    const UfoVoidMap& map1, const UfoVoidMap& map2, const lt_mapping::EvidenceParams& params)
{
    auto static_map(new pcl::PointCloud<pcl::PointXYZI>());
    static_map->reserve(first_map->points.size() + second_map->points.size());

    std::size_t dropped_first_dynamic = 0;
    std::size_t dropped_second_dynamic = 0;
    std::size_t nd_conflict_with_second = 0;

    // 진단 전용: 제거된 점의 고유 voxel 수와 ND 우선 제거로 가려진 self-void 수
    std::unordered_set<int64_t> dropped_first_dynamic_vox;
    std::unordered_set<int64_t> dropped_second_dynamic_vox;
    std::size_t first_nd_dropped = 0;
    std::size_t first_nd_and_selfvoid = 0;

    for (const auto& p : first_map->points) {
        int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, params.res);
        if (cls.nd_voxels.find(key) != cls.nd_voxels.end()) {
            first_nd_dropped++;
            if (map1.seenFree(ufo::Point(p.x, p.y, p.z))) {
                first_nd_and_selfvoid++;
            }
            continue;
        }
        if (map1.seenFree(ufo::Point(p.x, p.y, p.z))) {
            dropped_first_dynamic++;
            dropped_first_dynamic_vox.insert(key);
            continue;
        }
        static_map->points.push_back(p);
    }

    for (const auto& p : second_map->points) {
        int64_t key = lt_mapping::voxelKey(p.x, p.y, p.z, params.res);
        if (map2.seenFree(ufo::Point(p.x, p.y, p.z))) {
            dropped_second_dynamic++;
            dropped_second_dynamic_vox.insert(key);
            continue;
        }
        if (cls.nd_voxels.find(key) != cls.nd_voxels.end()) {
            nd_conflict_with_second++;
        }
        // 보수적 유지 원칙: 2세션 점은 명시적 폐기 근거가 없으면 유지
        static_map->points.push_back(p);
    }

    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Compose filter: drop_self_dynamic first=%zu second=%zu, nd_conflict_second=%zu",
                dropped_first_dynamic, dropped_second_dynamic, nd_conflict_with_second);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Compose filter detail: drop_self_dynamic_vox first=%zu second=%zu | "
                "first_nd_dropped=%zu (of which self-void=%zu)",
                dropped_first_dynamic_vox.size(), dropped_second_dynamic_vox.size(),
                first_nd_dropped, first_nd_and_selfvoid);

    pcl::VoxelGrid<pcl::PointXYZI> down_size_filter;
    down_size_filter.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
    down_size_filter.setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr(static_map));

    auto final_map(new pcl::PointCloud<pcl::PointXYZI>());
    down_size_filter.filter(*final_map);
    final_map->width = final_map->points.size();
    final_map->height = 1;
    final_map->is_dense = false;
    return pcl::PointCloud<pcl::PointXYZI>::Ptr(final_map);
}

}  // namespace

void MapUpdate()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr first_map_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr second_map_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::io::loadPCDFile(save_directory + "FirstMap.pcd", *first_map_cloud);
    pcl::io::loadPCDFile(save_directory + "SecondMap.pcd", *second_map_cloud);

    std::vector<int> idx1, idx2;
    pcl::removeNaNFromPointCloud(*first_map_cloud, *first_map_cloud, idx1);
    pcl::removeNaNFromPointCloud(*second_map_cloud, *second_map_cloud, idx2);

    auto session1 = buildVoidMap(1, FirstMapSize, persistence_params);
    auto session2 = buildVoidMap(2, SecondMapSize, persistence_params);

    int s1_void = 0, s1_hit = 0;
    int s2_void = 0, s2_hit = 0;
    computeMapStateStats(session1.map, first_map_cloud, persistence_params.res, s1_void, s1_hit);
    computeMapStateStats(session2.map, second_map_cloud, persistence_params.res, s2_void, s2_hit);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Session1 map stats: void=%d, hit=%d | Session2 map stats: void=%d, hit=%d",
                s1_void, s1_hit, s2_void, s2_hit);

    auto is_1in2 = buildInterestSet(first_map_cloud, persistence_params.res);
    auto is_2in1 = buildInterestSet(second_map_cloud, persistence_params.res);

    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Input map density: first points=%zu voxels(res=%.2f)=%zu (%.3f pts/vox) | "
                "second points=%zu voxels=%zu (%.3f pts/vox) | VOXEL_SIZE=%.2f",
                first_map_cloud->size(), persistence_params.res, is_1in2.size(),
                static_cast<double>(first_map_cloud->size()) /
                    static_cast<double>(std::max<size_t>(is_1in2.size(), 1)),
                second_map_cloud->size(), is_2in1.size(),
                static_cast<double>(second_map_cloud->size()) /
                    static_cast<double>(std::max<size_t>(is_2in1.size(), 1)),
                VOXEL_SIZE);

    std::unordered_map<int64_t, lt_mapping::VoidEvidence> ev_1in2;
    std::unordered_map<int64_t, lt_mapping::VoidEvidence> ev_2in1;

    accumulateEvidence(session2.frames, is_1in2, ev_1in2, persistence_params);
    accumulateEvidence(session1.frames, is_2in1, ev_2in1, persistence_params);

    auto summarizeEvidence = [](const std::unordered_map<int64_t, lt_mapping::VoidEvidence>& ev,
                                const char* tag) {
        double q_sum = 0.0;
        double kf_sum = 0.0;
        // DDA ray가 통과하지 않은 빈 엔트리(q_sum<=0)는 제외해 기존 지표 의미를 유지한다.
        size_t evidenced = 0;
        for (const auto& kv : ev) {
            if (kv.second.q_sum <= 0.0f) {
                continue;
            }
            evidenced++;
            q_sum += kv.second.q_sum;
            kf_sum += kv.second.n_keyframes;
        }
        double cnt = static_cast<double>(std::max<size_t>(evidenced, 1));
        RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                    "%s evidence: interest=%zu, avg_q_sum=%.4f, avg_n_keyframes=%.4f",
                    tag, evidenced, q_sum / cnt, kf_sum / cnt);
    };
    summarizeEvidence(ev_1in2, "1in2");
    summarizeEvidence(ev_2in1, "2in1");

    auto cls = classifyChanges(first_map_cloud, second_map_cloud, session1.map, session2.map, ev_1in2,
                               ev_2in1, persistence_params);

    // [연결 성분 후처리]
    // 디버그 pcd 저장과 composeFinalMap() 호출 이전에 적용해야 ND.pcd/PD.pcd 와 StaticMap.pcd 가
    // 같은 판정을 담는다. UE 집합에는 적용하지 않는다(UE 는 이미 62~72% 가 31 점 이상 대형 군집이고,
    // UE voxel 은 애초에 최종 지도에서 제거되지 않는다).
    const std::size_t nd_before_filter = cls.nd_voxels.size();
    const std::size_t pd_before_filter = cls.pd_voxels.size();
    auto nd_cf = filterSmallClusters(cls.nd_voxels, cls.nd_cloud, persistence_params.cluster_eps,
                                     persistence_params.min_cluster_size, persistence_params.res);
    auto pd_cf = filterSmallClusters(cls.pd_voxels, cls.pd_cloud, persistence_params.cluster_eps,
                                     persistence_params.min_cluster_size, persistence_params.res);

    auto clusterHistLine = [](const char* tag, const ClusterFilterStats& s) {
        std::ostringstream os;
        os << "Cluster size histogram (pre-filter) " << tag << ": clusters=" << s.clusters_total;
        for (std::size_t i = 0; i < kClusterBinCount; ++i) {
            os << " [" << clusterBinLabel(i) << "]=" << s.cluster_hist[i] << "c/" << s.point_hist[i]
               << "p";
        }
        return os.str();
    };
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"), "%s",
                clusterHistLine("ND", nd_cf).c_str());
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"), "%s",
                clusterHistLine("PD", pd_cf).c_str());
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Cluster filter (eps=%.2f, min=%d, applied=%s): "
                "ND %zu -> %zu points, %zu -> %zu voxels (removed %zu points in %zu/%zu clusters) | "
                "PD %zu -> %zu points, %zu -> %zu voxels (removed %zu points in %zu/%zu clusters) | "
                "elapsed=%.3f sec",
                persistence_params.cluster_eps, persistence_params.min_cluster_size,
                (nd_cf.applied || pd_cf.applied) ? "yes" : "no (baseline)", nd_cf.points_before,
                nd_cf.points_after, nd_cf.voxels_before, nd_cf.voxels_after, nd_cf.points_removed,
                nd_cf.clusters_removed, nd_cf.clusters_total, pd_cf.points_before,
                pd_cf.points_after, pd_cf.voxels_before, pd_cf.voxels_after, pd_cf.points_removed,
                pd_cf.clusters_removed, pd_cf.clusters_total,
                nd_cf.elapsed_sec + pd_cf.elapsed_sec);
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "ND/PD yield (after cluster filter): ND/void2=%zu/%zu (%.2f%%), "
                "PD/void1=%zu/%zu (%.2f%%) | pre-filter ND=%zu PD=%zu",
                cls.nd_voxels.size(), cls.void2_total,
                cls.void2_total == 0 ? 0.0
                                     : 100.0 * static_cast<double>(cls.nd_voxels.size()) /
                                           static_cast<double>(cls.void2_total),
                cls.pd_voxels.size(), cls.void1_total,
                cls.void1_total == 0 ? 0.0
                                     : 100.0 * static_cast<double>(cls.pd_voxels.size()) /
                                           static_cast<double>(cls.void1_total),
                nd_before_filter, pd_before_filter);

    if (!cls.first_ue->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "FirstUE.pcd", *cls.first_ue);
    }
    if (!cls.second_ue->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "SecondUE.pcd", *cls.second_ue);
    }
    if (!cls.nd_cloud->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "ND.pcd", *cls.nd_cloud);
    }
    if (!cls.pd_cloud->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "PD.pcd", *cls.pd_cloud);
    }
    if (!cls.persistence_cloud->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "PersistenceMap.pcd", *cls.persistence_cloud);
    }
    if (!cls.evidence_debug_cloud->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "EvidenceDebug.pcd", *cls.evidence_debug_cloud);
    }

    auto static_map_cloud = composeFinalMap(first_map_cloud, second_map_cloud, cls, session1.map,
                                            session2.map, persistence_params);

    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "Point classification: ND=%zu(vox=%zu), PD=%zu(vox=%zu), UE1=%zu(vox=%zu), UE2=%zu(vox=%zu), static=%zu",
                cls.nd_cloud->size(), cls.nd_voxels.size(), cls.pd_cloud->size(), cls.pd_voxels.size(),
                cls.first_ue->size(), cls.first_ue_voxels.size(), cls.second_ue->size(), cls.second_ue_voxels.size(),
                static_map_cloud->size());
    RCLCPP_INFO(rclcpp::get_logger("long_term_mapping"),
                "PD diagnostic: pd_voxels=%zu, pd_points=%zu",
                cls.pd_voxels.size(), cls.pd_cloud->size());

    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*static_map_cloud, map_msg);
    map_msg.header.frame_id = "map";
    PubMerge_map->publish(map_msg);
    pcl::io::savePCDFileBinary(save_directory + "StaticMap.pcd", *static_map_cloud);
}

void saveEdges()
{
    for (int k = 0; k < FirstMapEdges.size(); k++)
    {
        auto edge = FirstMapEdges[k];
        int idx = 0;
        saveEdge(edge, idx);        
    }

    for (int k = 0; k < SecondMapEdges.size(); k++)
    {
        auto edge = SecondMapEdges[k];
        int idx = FirstMapSize;
        saveEdge(edge, idx);        
    }
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto nh = rclcpp::Node::make_shared("LTmapping");    

    // QoS for visualization topics (latest data only)
    auto qos_viz = rclcpp::QoS(rclcpp::KeepLast(1));
    qos_viz.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_viz.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
    
    // 시각화 데이터는 최신 데이터만
    First_kf_node_pub = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/first_kf_node", qos_viz);
    Second_kf_node_pub = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/second_kf_node", qos_viz);
    Merge_kf_node_pub = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/merge_kf_node", qos_viz);
    LoopLineMarker_pub = nh->create_publisher<visualization_msgs::msg::Marker>("/loopLine", qos_viz);
    PubFirstMap_path = nh->create_publisher<nav_msgs::msg::Path>("/First_path", qos_viz);
    PubSecondMap_path = nh->create_publisher<nav_msgs::msg::Path>("/Second_path", qos_viz);
    PubMerge_path = nh->create_publisher<nav_msgs::msg::Path>("/Merge_path", qos_viz);
    PubMerge_map = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/Merge_map", qos_viz);
    completion_pub = nh->create_publisher<std_msgs::msg::Bool>("/lt_mapping_complete", qos_viz);
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    setParams(nh);
    
    getDirectory();

    if (!loadFiles())   return -1;
    
    initNoises();

    getEdges();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Session Edge Loading Complete.");

    placeRecognition();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Place Recognition Complete.");
    
    getLoopEdges();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Loop Edge Generation Complete. size : %d", loop_pairs.size());

    getPoses();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Pose Factor loading Complete.");

    runISAM2opt();

    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Graph Optimization Complete.");

    generateOptimizedMap();
    
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Map Merging Complete.");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    MapUpdate();

    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Map Update Complete.");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    saveEdges();
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Long Term SLAM Complete.");

    optimized_stream.close();
    edge_stream.close();

    // Publish completion message
    std_msgs::msg::Bool completion_msg;
    completion_msg.data = true;
    completion_pub->publish(completion_msg);
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "Completion message published.");

    // Allow time for message to be sent
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Clean up publishers before shutting down ROS2
    completion_pub.reset();
    First_kf_node_pub.reset();
    Second_kf_node_pub.reset();
    Merge_kf_node_pub.reset();
    LoopLineMarker_pub.reset();
    PubFirstMap_path.reset();
    PubSecondMap_path.reset();
    PubMerge_path.reset();
    PubMerge_map.reset();

    // Clean up ISAM2 object
    if (isam)
    {
        delete isam;
        isam = nullptr;
    }

    // Allow time for proper cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    rclcpp::shutdown();
    return 0;
}