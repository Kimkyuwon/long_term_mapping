#include <mutex>
#include <math.h>
#include <thread>
#include <queue>
#include <fstream>
#include <csignal>
#include <optional>
#include <chrono>
#include <sstream>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
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
#include <pcl/console/print.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
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
double loop_search_radius;
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
    nh->declare_parameter("loop_search_radius", 15.0);
    nh->declare_parameter("directory1", std::string(""));
    nh->declare_parameter("directory2", std::string(""));
    nh->declare_parameter("output_directory", std::string(""));

    nh->declare_parameter("persistence.res", 0.2);

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
    nh->get_parameter_or<double>("loop_search_radius", loop_search_radius, 15.0);

    nh->get_parameter_or<float>("persistence.res", persistence_params.res, 0.2f);

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
    gicp.setNumThreads(4);
    gicp.setCorrespondenceRandomness(15);
    gicp.setMaximumIterations(3);
    gicp.setTransformationEpsilon(0.01);
    gicp.setEuclideanFitnessEpsilon(0.01);

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

    if (dop_ratio < dop_thres && matching_dop < 1.0)
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

    // FirstMap pose KD-tree for geometric nearest-node loop candidates
    pcl::PointCloud<pcl::PointXYZI>::Ptr first_pose_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    first_pose_cloud->reserve(FirstMapSize);
    for (int k = 0; k < FirstMapSize; ++k)
    {
        pcl::PointXYZI p;
        p.x = FirstMapPoses[k].x;
        p.y = FirstMapPoses[k].y;
        p.z = FirstMapPoses[k].z;
        p.intensity = static_cast<float>(k);
        first_pose_cloud->push_back(p);
    }
    kdtree->setInputCloud(first_pose_cloud);

    const float search_radius_sq = static_cast<float>(loop_search_radius * loop_search_radius);

    for (int i = 0; i < SecondMapSize; ++i)
    {
        gtsam::Pose3 pose2 = Pose6toGTSAMPose3(SecondMapPoses[i]);
        Eigen::Matrix4d pose2_anchored_TF = A2_anchor.matrix() * pose2.matrix();
        gtsam::Pose3 pose2_anchored(pose2_anchored_TF);

        pcl::PointXYZI query;
        query.x = pose2_anchored.translation().x();
        query.y = pose2_anchored.translation().y();
        query.z = pose2_anchored.translation().z();

        std::vector<int> nn_idx;
        std::vector<float> nn_sqdist;
        if (kdtree->nearestKSearch(query, 1, nn_idx, nn_sqdist) == 0)
            continue;
        if (nn_sqdist[0] > search_radius_sq)
            continue;

        const int prev_node_idx = nn_idx[0];
        const int curr_node_idx = i;

        pcl::PointCloud<pcl::PointXYZI>::Ptr FirstScanCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(dir1_scans_path + std::to_string(prev_node_idx) + ".pcd", *FirstScanCloud);
        pcl::PointCloud<pcl::PointXYZI>::Ptr SecondScanCloud(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(dir2_scans_path + std::to_string(curr_node_idx) + ".pcd", *SecondScanCloud);
        std::vector<int> scan_first_indices;
        std::vector<int> scan_second_indices;
        pcl::removeNaNFromPointCloud(*FirstScanCloud, *FirstScanCloud, scan_first_indices);
        pcl::removeNaNFromPointCloud(*SecondScanCloud, *SecondScanCloud, scan_second_indices);

        Eigen::Matrix4f from_TF = get_TF_Matrix(FirstMapPoses[prev_node_idx]);
        Eigen::Matrix4f to_TF = A2_anchor.matrix().cast<float>() * get_TF_Matrix(SecondMapPoses[curr_node_idx]);
        Eigen::Matrix4f delta_TF = from_TF.inverse() * to_TF;
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
        Eigen::Vector3d delta_vec(delta_TF(0, 3), delta_TF(1, 3), delta_TF(2, 3));
        double matching_dop = computeDOP(MatchingCloud, delta_vec);
        if (matching_dop > 1.2)
            continue;

        auto relative_pose_optional = doGICPVirtualRelative(prev_node_idx, curr_node_idx, delta_TF);

        if (relative_pose_optional)
        {
            gtsam::Pose3 relative_pose = relative_pose_optional.value();
            gtSAMgraph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                getGlobalNodeIdx(1, prev_node_idx), getGlobalNodeIdx(2, curr_node_idx), relative_pose, robustLoopNoise));

            edge_stream << prev_node_idx << " " << (curr_node_idx + FirstMapSize) << " "
                << relative_pose.translation().x() << " " << relative_pose.translation().y() << " "
                << relative_pose.translation().z() << " " << relative_pose.rotation().roll() << " "
                << relative_pose.rotation().pitch() << " " << relative_pose.rotation().yaw() << " "
                << robustNoiseVector6(0) << " " << robustNoiseVector6(1) << " " << robustNoiseVector6(2) << " "
                << robustNoiseVector6(3) << " " << robustNoiseVector6(4) << " " << robustNoiseVector6(5) << endl;
            pair<int, int> loop_pair = make_pair(prev_node_idx, curr_node_idx);
            loop_pairs.push_back(loop_pair);
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


void MapUpdate()
{
    // full map: StaticMap compose / self-void 제거용
    pcl::PointCloud<pcl::PointXYZI>::Ptr first_full(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr second_full(new pcl::PointCloud<pcl::PointXYZI>());
    // nonground: PD/ND/UE 판정용 (지면 오검출 억제)
    pcl::PointCloud<pcl::PointXYZI>::Ptr first_nonground(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr second_nonground(new pcl::PointCloud<pcl::PointXYZI>());

    pcl::io::loadPCDFile(save_directory + "FirstMap.pcd", *first_full);
    pcl::io::loadPCDFile(save_directory + "SecondMap.pcd", *second_full);
    pcl::io::loadPCDFile(save_directory + "FirstNonGroundMap.pcd", *first_nonground);
    pcl::io::loadPCDFile(save_directory + "SecondNonGroundMap.pcd", *second_nonground);

    std::vector<int> idx;
    pcl::removeNaNFromPointCloud(*first_full, *first_full, idx);
    pcl::removeNaNFromPointCloud(*second_full, *second_full, idx);
    pcl::removeNaNFromPointCloud(*first_nonground, *first_nonground, idx);
    pcl::removeNaNFromPointCloud(*second_nonground, *second_nonground, idx);

    auto make_loader = [](int session_id) {
        return [session_id](int local_idx, Eigen::Vector3f& origin) {
            Pose6 key_pose;
            int global_key = getGlobalNodeIdx(session_id, local_idx);
            key_pose.x = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().x();
            key_pose.y = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().y();
            key_pose.z = isamCurrentEstimate.at<gtsam::Pose3>(global_key).translation().z();
            key_pose.roll = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().roll();
            key_pose.pitch = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().pitch();
            key_pose.yaw = isamCurrentEstimate.at<gtsam::Pose3>(global_key).rotation().yaw();

            Eigen::Matrix4f tf = createTransformMatrix(key_pose);
            origin = Eigen::Vector3f(tf(0, 3), tf(1, 3), tf(2, 3));

            std::string scan_path = (session_id == 1 ? dir1_scans_path : dir2_scans_path) +
                                    std::to_string(local_idx) + ".pcd";
            auto cloud = loadPointCloud(scan_path);
            if (!cloud) {
                return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
            }
            std::vector<int> nan_idx;
            pcl::removeNaNFromPointCloud(*cloud, *cloud, nan_idx);
            pcl::transformPointCloud(*cloud, *cloud, tf);
            return cloud;
        };
    };

    auto session1 = lt_mapping::buildVoidMap(1, FirstMapSize, persistence_params,
                                             static_cast<float>(MAX_DISTANCE), make_loader(1));
    auto session2 = lt_mapping::buildVoidMap(2, SecondMapSize, persistence_params,
                                             static_cast<float>(MAX_DISTANCE), make_loader(2));

    auto result = lt_mapping::detectAndCompose(
        first_nonground, second_nonground, first_full, second_full, session1, session2,
        persistence_params, static_cast<float>(blind), static_cast<float>(MAX_DISTANCE),
        static_cast<float>(VOXEL_SIZE));

    if (!result.cls.first_ue->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "FirstUE.pcd", *result.cls.first_ue);
    }
    if (!result.cls.second_ue->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "SecondUE.pcd", *result.cls.second_ue);
    }
    if (!result.cls.nd_cloud->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "ND.pcd", *result.cls.nd_cloud);
    }
    if (!result.cls.pd_cloud->empty()) {
        pcl::io::savePCDFileBinary(DebugDirectory + "PD.pcd", *result.cls.pd_cloud);
    }

    sensor_msgs::msg::PointCloud2 map_msg;
    pcl::toROSMsg(*result.static_map, map_msg);
    map_msg.header.frame_id = "map";
    PubMerge_map->publish(map_msg);
    pcl::io::savePCDFileBinary(save_directory + "StaticMap.pcd", *result.static_map);
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