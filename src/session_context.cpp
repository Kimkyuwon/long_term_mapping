#include "session_context.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <pcl/io/pcd_io.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>

#include "map_merge.hpp"

using namespace std;
namespace fs = std::filesystem;

// ---- P: 경로·스트림 ----
string save_directory, DebugDirectory, ScanDirectory, directory1, directory2, output_directory;

fstream optimized_stream, edge_stream;

// ---- S: 세션 데이터 ----
// sessions[0] = 세션1(구 First*/dir1_*), sessions[1] = 세션2(구 Second*/dir2_*).
std::vector<SessionData> sessions(2);

// ---- A: 알고리즘 파라미터 ----
double FOV_u, VOXEL_SIZE;
int MAX_DISTANCE;
double blind;
double dop_thres = 0;
double anchor_resolution;
double loop_search_radius;
lt_mapping::EvidenceParams persistence_params;
float session_rigid_residual_thres = -1.0f;
float rebuild_pose_shift_thres = 0.0f;

// initNoises()(map_merge.cpp) 하드코딩 값을 그대로 옮긴 것 — 값 변경 없음(Phase 1.1).
double noise_prior_rot_variance = 1e-12;
double noise_prior_trans_variance = 1e-12;
double noise_large_rot_variance = M_PI * M_PI;
double noise_large_trans_variance = 1e8;
double noise_odom_rot_variance = 1e-3;
double noise_odom_trans_variance = 1e-2;

// gicp/loopLine/isam은 모듈 C(map_merge) 소유(G/V 그룹) — 선언은 map_merge.hpp에서 온다.
// "생성 시점" 초기화만 setParams()에 남아 있는 전이 상태이며, Phase1 5단계(선택)에서
// 모듈 C의 초기화 함수로 옮긴다. 03.task_plan_rules.mdc 「Phase 1」 5단계 참고.

void setParams (std::shared_ptr<rclcpp::Node> nh)
{

    // PatchworkppGroundSeg.reset(new PatchWorkpp<pcl::PointXYZI>());

    nh->declare_parameter("blind", 0.01);
    nh->declare_parameter("fov_u", 2.0);
    nh->declare_parameter("max_distance", 80);
    nh->declare_parameter("anchor_resolution", 1.0);
    nh->declare_parameter("voxel_size", 0.4);
    nh->declare_parameter("dop_thres", 0.5);
    nh->declare_parameter("loop_search_radius", 15.0);
    nh->declare_parameter("directory1", std::string(""));
    nh->declare_parameter("directory2", std::string(""));
    nh->declare_parameter("output_directory", std::string(""));

    nh->declare_parameter("persistence.res", 0.2);
    nh->declare_parameter("evidence.res", 0.2);
    nh->declare_parameter("persistence.session_rigid_residual_thres", -1.0);
    nh->declare_parameter("persistence.rebuild_pose_shift_thres", 0.0);

    nh->declare_parameter("noise.prior_rot_variance", noise_prior_rot_variance);
    nh->declare_parameter("noise.prior_trans_variance", noise_prior_trans_variance);
    nh->declare_parameter("noise.large_rot_variance", noise_large_rot_variance);
    nh->declare_parameter("noise.large_trans_variance", noise_large_trans_variance);
    nh->declare_parameter("noise.odom_rot_variance", noise_odom_rot_variance);
    nh->declare_parameter("noise.odom_trans_variance", noise_odom_trans_variance);

    nh->get_parameter_or<double>("blind", blind, 0.01);
    nh->get_parameter_or<double>("fov_u", FOV_u, 2.0);
    nh->get_parameter_or<int>("max_distance", MAX_DISTANCE, 80);
    nh->get_parameter_or<double>("anchor_resolution", anchor_resolution, 1.0);
    nh->get_parameter_or<double>("voxel_size", VOXEL_SIZE, 0.4);
    nh->get_parameter_or<double>("dop_thres", dop_thres, 0.5);
    nh->get_parameter_or<double>("loop_search_radius", loop_search_radius, 15.0);

    nh->get_parameter_or<float>("evidence.res", persistence_params.res, 0.2f);
    nh->get_parameter_or<float>("persistence.res", persistence_params.res, persistence_params.res);
    nh->get_parameter_or<float>("persistence.session_rigid_residual_thres", session_rigid_residual_thres, -1.0f);
    nh->get_parameter_or<float>("persistence.rebuild_pose_shift_thres", rebuild_pose_shift_thres, 0.0f);
    if (session_rigid_residual_thres < 0.0f) {
        session_rigid_residual_thres = persistence_params.res * 0.25f;
    }

    nh->get_parameter_or<double>("noise.prior_rot_variance", noise_prior_rot_variance, 1e-12);
    nh->get_parameter_or<double>("noise.prior_trans_variance", noise_prior_trans_variance, 1e-12);
    nh->get_parameter_or<double>("noise.large_rot_variance", noise_large_rot_variance, M_PI * M_PI);
    nh->get_parameter_or<double>("noise.large_trans_variance", noise_large_trans_variance, 1e8);
    nh->get_parameter_or<double>("noise.odom_rot_variance", noise_odom_rot_variance, 1e-3);
    nh->get_parameter_or<double>("noise.odom_trans_variance", noise_odom_trans_variance, 1e-2);

    // 먼저 ROS2 파라미터에서 값 확인
    nh->get_parameter_or<std::string>("directory1", directory1, std::string(""));
    nh->get_parameter_or<std::string>("directory2", directory2, std::string(""));
    nh->get_parameter_or<std::string>("output_directory", output_directory, std::string(""));

    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
        "[000.010 Params] directory1=%s directory2=%s output_directory=%s evidence.res=%.3f "
        "persistence.session_rigid_residual_thres=%.3f persistence.rebuild_pose_shift_thres=%.3f",
        directory1.c_str(), directory2.c_str(), output_directory.c_str(),
        persistence_params.res, session_rigid_residual_thres, rebuild_pose_shift_thres);
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
        "[000.010 Params] noise.prior_rot_variance=%.3g noise.prior_trans_variance=%.3g "
        "noise.large_rot_variance=%.3g noise.large_trans_variance=%.3g "
        "noise.odom_rot_variance=%.3g noise.odom_trans_variance=%.3g",
        noise_prior_rot_variance, noise_prior_trans_variance,
        noise_large_rot_variance, noise_large_trans_variance,
        noise_odom_rot_variance, noise_odom_trans_variance);

    // GICP·ISAM2 튜닝 값은 의도적으로 ROS 파라미터 미노출 상태로 남긴다 — 이 값들을 옮기는 작업은
    // Phase 1 5단계(선택, gicp/isam 이관)의 범위이며 03.task_plan_rules.mdc 「범위 결정」에서
    // 현재 단계 제외로 명시됐다. 값은 원본과 동일하게 유지한다.
    gicp.setMaxCorrespondenceDistance(2.0);
    gicp.setNumThreads(4);
    gicp.setCorrespondenceRandomness(15);
    gicp.setMaximumIterations(10);
    gicp.setTransformationEpsilon(0.01);
    gicp.setEuclideanFitnessEpsilon(0.01);

    loopLine.type = visualization_msgs::msg::Marker::LINE_LIST;
    loopLine.action = visualization_msgs::msg::Marker::ADD;
    loopLine.color.b = 1.0; loopLine.color.a = 0.5;
    loopLine.scale.x = 0.1;
    loopLine.header.frame_id = "map";

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

}

pcl::PointCloud<pcl::PointXYZI>::Ptr loadPointCloud(const std::string& filepath)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    if (pcl::io::loadPCDFile(filepath, *cloud) == -1) {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] load_point_cloud_failed path=%s", filepath.c_str());
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr loadRawScanPointCloud(const std::string& scans_dir, int idx)
{
    const std::string raw_path = scans_dir + std::to_string(idx) + ".pcd";
    if (!fs::exists(raw_path)) {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"),
                     "[100.110 SessionLoad] raw_scan_missing path=%s", raw_path.c_str());
        return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    }
    return loadPointCloud(raw_path);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr loadProcessedScanPointCloud(const std::string& scans_dir, int idx)
{
    const std::string processed_path = scans_dir + std::to_string(idx) + "_remove.pcd";
    if (!fs::exists(processed_path)) {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"),
                     "[100.110 SessionLoad] processed_scan_missing path=%s", processed_path.c_str());
        return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
    }
    return loadPointCloud(processed_path);
}

void saveEdge(tuple<int, int, gtsam::Vector, Pose6> edge, int idx)
{
    const int prev_node_idx = get<0>(edge) + idx;
    const int curr_node_idx = get<1>(edge) + idx;
    gtsam::Vector edge_score = get<2>(edge);
    const Pose6 edge_pose = get<3>(edge);
    gtsam::noiseModel::Diagonal::shared_ptr EdgeNoise = gtsam::noiseModel::Diagonal::Variances(edge_score);
    gtsam::Pose3 relative_pose = Pose6toGTSAMPose3(edge_pose);
    edge_stream << prev_node_idx << " " << curr_node_idx << " " << relative_pose.translation().x() << " " << relative_pose.translation().y() << " " 
        << relative_pose.translation().z() << " " << relative_pose.rotation().roll() << " " << relative_pose.rotation().pitch() << " " 
        << relative_pose.rotation().yaw() << " " << edge_score(0) << " " << edge_score(1) << " " << edge_score(2) << " " << edge_score(3) << " " 
        << edge_score(4) << " " << edge_score(5) << endl;
}

namespace {

// output_directory 안전성 검사. 빈 값이거나 정규화 후 ROOT_DIR와 같거나 그 바깥 경로가
// 되면 즉시 예외를 던져 차단한다. 03.task_plan_rules.mdc 「0-A 상세」 항목 (a)(b)(f) 대응.
// git에서 되돌린 직후처럼 세 경로가 모두 빈 값인 상태로 실행해도 여기서 막힌다.
void validateDangerousPathOverlap(const std::string& output_dir_raw, const fs::path& root_dir)
{
    if (output_dir_raw.empty())
    {
        throw std::runtime_error(
            "[PathSafety] output_directory is empty. Refusing to write/delete under package source root: "
            + root_dir.string());
    }

    const fs::path candidate = root_dir / output_dir_raw;
    std::error_code ec;
    fs::path normalized_root = fs::weakly_canonical(root_dir, ec);
    if (ec) { normalized_root = root_dir.lexically_normal(); ec.clear(); }
    fs::path normalized_candidate = fs::weakly_canonical(candidate, ec);
    if (ec) { normalized_candidate = candidate.lexically_normal(); }

    if (normalized_candidate == normalized_root)
    {
        throw std::runtime_error(
            "[PathSafety] output_directory resolves to the package source root itself: "
            + normalized_candidate.string());
    }

    // normalized_candidate가 normalized_root의 하위 경로인지 확인한다.
    auto root_it = normalized_root.begin();
    auto cand_it = normalized_candidate.begin();
    for (; root_it != normalized_root.end() && cand_it != normalized_candidate.end(); ++root_it, ++cand_it)
    {
        if (*root_it != *cand_it)
        {
            throw std::runtime_error(
                "[PathSafety] output_directory escapes the package source root: "
                + normalized_candidate.string());
        }
    }
    if (root_it != normalized_root.end())
    {
        throw std::runtime_error(
            "[PathSafety] output_directory does not resolve under the package source root: "
            + normalized_candidate.string());
    }
}

// system("rm -r ...")를 대신한다. 반환값을 버리지 않고 실패 시 예외로 알린다.
// 대상이 원래 없던 경우(ENOENT 계열)는 정상으로 취급한다.
void removeDirectorySafely(const fs::path& target)
{
    std::error_code ec;
    fs::remove_all(target, ec);
    if (ec)
    {
        throw std::runtime_error(
            "[PathSafety] Failed to remove directory: " + target.string() + " (" + ec.message() + ")");
    }
}

}  // namespace

void getDirectory()
{
    // ROOT_DIR 경로 설정 (현재 프로젝트 디렉토리)
    std::string root_path = string(ROOT_DIR);
    fs::path root_dir(root_path);

    validateDangerousPathOverlap(output_directory, root_dir);

    save_directory = root_path + output_directory + "/";
    removeDirectorySafely(save_directory);
    fs::create_directories(save_directory);

    DebugDirectory = save_directory + "Debug/";
    removeDirectorySafely(DebugDirectory);
    fs::create_directories(DebugDirectory);
    
    ScanDirectory = save_directory + "Scans/";
    removeDirectorySafely(ScanDirectory);
    fs::create_directories(ScanDirectory);
    
    string optimized_path = save_directory + "/optimized_poses.txt";
    optimized_stream = std::fstream(optimized_path, std::fstream::out);
    optimized_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!optimized_stream) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] optimized_stream_open_failed path=%s", optimized_path.c_str());
    }

    string edge_directory = save_directory + "/edges.txt";
    edge_stream = std::fstream(edge_directory, std::fstream::out);
    edge_stream.precision(std::numeric_limits<double>::max_digits10);
    if (!edge_stream) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] edge_stream_open_failed path=%s", edge_directory.c_str());
    }
    
    // 첫 번째 디렉토리 경로들
    sessions[0].scans_path = directory1 + "/Scans/";
    sessions[0].poses_path = directory1 + "/optimized_poses.txt";
    sessions[0].edges_path = directory1 + "/edges.txt";
    sessions[0].map_path   = directory1 + "/StaticMap.pcd";
    
    // 두 번째 디렉토리 경로들
    sessions[1].scans_path = directory2 + "/Scans/";
    sessions[1].poses_path = directory2 + "/optimized_poses.txt";
    sessions[1].edges_path = directory2 + "/edges.txt";
    sessions[1].map_path   = directory2 + "/StaticMap.pcd";
    
    // 경로 출력으로 확인
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
        "[100.110 SessionLoad] session1_scans=%s session1_poses=%s session1_edges=%s "
        "session2_scans=%s session2_poses=%s session2_edges=%s",
        sessions[0].scans_path.c_str(), sessions[0].poses_path.c_str(), sessions[0].edges_path.c_str(),
        sessions[1].scans_path.c_str(), sessions[1].poses_path.c_str(), sessions[1].edges_path.c_str());
}

// 포즈 파일을 읽는 함수
bool loadPoseFile(const std::string& filepath, std::vector<double>& times, std::vector<Pose6>& poses)
{
    std::ifstream file(filepath);
    if (!file.is_open()) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] open_failed path=%s", filepath.c_str());
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
            RCLCPP_WARN(rclcpp::get_logger("LTmapping"),
                "[100.110 SessionLoad] parse_failed line=%d path=%s content=%s",
                line_count, filepath.c_str(), line.c_str());
            continue;  // 파싱 실패한 줄은 건너뛰고 계속 진행
        }
        
        // 시간을 벡터에 저장
        times.push_back(time);
        
        // Euler angles로 변환하여 Pose6로 저장
        Pose6 pose = poseToPose6(x, y, z, qx, qy, qz, qw);
        poses.push_back(pose);
    }
    
    file.close();
    
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
        "[100.110 SessionLoad] loaded poses=%zu path=%s", times.size(), filepath.c_str());
    return true;
}

// 엣지 파일을 읽는 함수
bool loadEdgeFile(const std::string& filepath, std::vector<tuple<int, int, gtsam::Vector, Pose6>>& edges)
{
    std::ifstream file(filepath);
    if (!file.is_open()) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] open_failed path=%s", filepath.c_str());
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
            RCLCPP_WARN(rclcpp::get_logger("LTmapping"),
                "[100.110 SessionLoad] parse_failed line=%d path=%s content=%s",
                line_count, filepath.c_str(), line.c_str());
            continue;  // 파싱 실패한 줄은 건너뛰고 계속 진행
        }
        gtsam::Vector edgeNoiseVector(6);
        edgeNoiseVector << covaricance1, covaricance2, covaricance3, covaricance4, covaricance5, covaricance6;
        tuple<int, int, gtsam::Vector, Pose6> edge = make_tuple(prev_idx, curr_idx, edgeNoiseVector, pose);
        
        edges.push_back(edge);
    }
    
    file.close();
    
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
        "[100.110 SessionLoad] loaded edges=%zu path=%s", edges.size(), filepath.c_str());
    return true;
}

bool loadFiles()
{
    // 포즈 파일들 읽기
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] loading_pose_files_begin");

    // 첫 번째 맵의 포즈 파일 읽기
    if (!loadPoseFile(sessions[0].poses_path, sessions[0].time, sessions[0].poses)) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] load_failed session=1 kind=poses");
        return false;
    }
    
    // 첫 번째 맵의 엣지 파일 읽기
    if (!loadEdgeFile(sessions[0].edges_path, sessions[0].edges)) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] load_failed session=1 kind=edges");
        return false;
    }

    // 두 번째 맵의 포즈 파일 읽기
    if (!loadPoseFile(sessions[1].poses_path, sessions[1].time, sessions[1].poses)) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] load_failed session=2 kind=poses");
        return false;
    }
    
    // 두 번째 맵의 엣지 파일 읽기
    if (!loadEdgeFile(sessions[1].edges_path, sessions[1].edges)) 
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "[100.110 SessionLoad] load_failed session=2 kind=edges");
        return false;
    }

    sessions[0].size = sessions[0].poses.size();
    sessions[1].size = sessions[1].poses.size();
    return true;
}

void saveEdges()
{
    for (int k = 0; k < sessions[0].edges.size(); k++)
    {
        auto edge = sessions[0].edges[k];
        int idx = 0;
        saveEdge(edge, idx);        
    }

    for (int k = 0; k < sessions[1].edges.size(); k++)
    {
        auto edge = sessions[1].edges[k];
        int idx = sessions[0].size;
        saveEdge(edge, idx);        
    }
}
