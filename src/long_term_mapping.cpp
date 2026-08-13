#include <chrono>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <pcl/console/print.h>

#include "lt_common.hpp"
#include "map_merge.hpp"
#include "session_context.hpp"
#include "voxel_evidence.hpp"

using namespace std;

// long_term_mapping.cpp — 진입점. main() + 오케스트레이션만 둔다.
// 01.developerules.mdc 「코드 구조 규칙 (모듈 경계)」 참고. 모듈 A~D는
// lt_common.hpp / session_context.* / map_merge.* / voxel_evidence.*에 있다.


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto nh = rclcpp::Node::make_shared("LTmapping");    

    // QoS for visualization topics (latest data only)
    auto qos_viz = rclcpp::QoS(rclcpp::KeepLast(1));
    qos_viz.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    qos_viz.durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);
    
    // 시각화 데이터는 최신 데이터만
    initMapMergePublishers(nh, qos_viz);
    initMergeMapPublisher(nh, qos_viz);
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

    setParams(nh);
    
    try
    {
        getDirectory();
    }
    catch (const std::runtime_error& e)
    {
        RCLCPP_ERROR(rclcpp::get_logger("LTmapping"), "%s", e.what());
        // 전역 퍼블리셔가 rclcpp::shutdown() 이후까지 살아있으면 정리 시점에 rmw 오류가 난다.
        // 정상 종료 경로(main 하단)와 동일하게 shutdown 전에 명시적으로 reset한다.
        resetMapMergePublishers();
        resetMergeMapPublisher();
        rclcpp::shutdown();
        return -1;
    }

    if (!loadFiles())   return -1;
    
    initNoises();

    getEdges();

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[100.180 EdgeLoad] status=done");

    getLoopEdges();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    if (loop_pairs.empty())
    {
        RCLCPP_WARN(rclcpp::get_logger("LTmapping"),
            "[100.130 LoopAccept] accepted=0 dop_thres=%.2f", dop_thres);
    }
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[100.170 LoopEdges] loop_pairs=%zu", loop_pairs.size());
    logLoopGicpDiagnostics();

    getPoses();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[100.180 PoseFactors] status=done");

    runISAM2opt();

    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[100.180 GraphOpt] status=done");

    const auto first_optimized = collectOptimizedSessionPoses(1, sessions[0].size);
    const auto second_optimized = collectOptimizedSessionPoses(2, sessions[1].size);
    const auto first_stats = evaluateSessionRigidity(1, sessions[0].poses, first_optimized);
    const auto second_stats = evaluateSessionRigidity(2, sessions[1].poses, second_optimized);
    logSessionRigidityStats(first_stats, session_rigid_residual_thres, rebuild_pose_shift_thres);
    logSessionRigidityStats(second_stats, session_rigid_residual_thres, rebuild_pose_shift_thres);

    generateOptimizedMap();
    
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[100.500 MapOutput] status=done");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    MapUpdate();

    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[400.460 MapUpdate] status=done");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    saveEdges();
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[100.500 SaveEdges] status=done");

    optimized_stream.close();
    edge_stream.close();

    // Publish completion message
    std_msgs::msg::Bool completion_msg;
    completion_msg.data = true;
    completion_pub->publish(completion_msg);
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"), "[000.090 Complete] completion_msg_published=true");

    // Allow time for message to be sent
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Clean up publishers before shutting down ROS2
    resetMapMergePublishers();
    resetMergeMapPublisher();

    // Clean up ISAM2 object
    cleanupIsam();

    // Allow time for proper cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    rclcpp::shutdown();
    return 0;
}