#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace lt_mapping {

// 3D voxel index -> int64 키 (각 축 21bit, 부호 포함)
inline int64_t voxelKey(float x, float y, float z, float res)
{
    int64_t ix = (int64_t)std::floor(x / res) & 0x1FFFFF;
    int64_t iy = (int64_t)std::floor(y / res) & 0x1FFFFF;
    int64_t iz = (int64_t)std::floor(z / res) & 0x1FFFFF;
    return ix | (iy << 21) | (iz << 42);
}

inline int64_t voxelKeyFromIndex(int ix, int iy, int iz)
{
    int64_t x = static_cast<int64_t>(ix) & 0x1FFFFF;
    int64_t y = static_cast<int64_t>(iy) & 0x1FFFFF;
    int64_t z = static_cast<int64_t>(iz) & 0x1FFFFF;
    return x | (y << 21) | (z << 42);
}

struct VoidEvidence {
    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();
    float q_sum = 0.f;
    uint16_t n_rays = 0;
    uint16_t n_keyframes = 0;
    int32_t last_kf = -1;
};

inline float effectiveCount(float n, float n_sat)
{
    const float s = std::max(n_sat, 1e-3f);
    return s * (1.0f - std::exp(-std::max(n, 0.0f) / s));
}

struct Persistence {
    float log_odds = 0.f;
    uint16_t hit_cnt = 0;
    float void_w_sum = 0.f;
};

// ROS 로는 persistence.res 만 노출. 나머지는 코드 기본값.
struct EvidenceParams {
    float res = 0.2f;
    float r0 = 15.f;
    float r_s = 20.f;
    float d_surf = 0.5f;
    float aniso_typ = 0.3f;
    float n_req = 3.f;
    float w_min = 0.05f;
    float n_sat = 5.f;
    float l_hit = 0.85f;
    float l_void = 0.40f;
    float l_max = 5.f;
    float tau_del = 0.30f;
    float tau_add = 0.70f;
    int cross_dilate = 1;
    float dilate_weight = 0.5f;
};

struct FrameData {
    int kf_id = 0;
    Eigen::Vector3f origin = Eigen::Vector3f::Zero();
    pcl::PointCloud<pcl::PointXYZI>::Ptr world_cloud;
};

// UFOMap 타입은 .cpp 에만 두어 헤더 ODR(multiple definition)을 피한다.
class SessionMaps {
public:
    explicit SessionMaps(float res);
    ~SessionMaps();
    SessionMaps(SessionMaps&&) noexcept;
    SessionMaps& operator=(SessionMaps&&) noexcept;
    SessionMaps(const SessionMaps&) = delete;
    SessionMaps& operator=(const SessionMaps&) = delete;

    struct Impl;
    Impl& impl() { return *impl_; }
    const Impl& impl() const { return *impl_; }

private:
    std::unique_ptr<Impl> impl_;
};

struct ClassificationResult {
    pcl::PointCloud<pcl::PointXYZI>::Ptr first_ue{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::PointCloud<pcl::PointXYZI>::Ptr second_ue{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::PointCloud<pcl::PointXYZI>::Ptr nd_cloud{new pcl::PointCloud<pcl::PointXYZI>()};
    pcl::PointCloud<pcl::PointXYZI>::Ptr pd_cloud{new pcl::PointCloud<pcl::PointXYZI>()};
    std::unordered_set<int64_t> first_ue_voxels;
    std::unordered_set<int64_t> second_ue_voxels;
    std::unordered_set<int64_t> nd_voxels;
    std::unordered_set<int64_t> pd_voxels;
};

struct MapUpdateResult {
    ClassificationResult cls;
    pcl::PointCloud<pcl::PointXYZI>::Ptr static_map{new pcl::PointCloud<pcl::PointXYZI>()};
};

inline float clampLogOdds(float l, float lim)
{
    return std::max(-lim, std::min(lim, l));
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

using WorldScanLoader =
    std::function<pcl::PointCloud<pcl::PointXYZI>::Ptr(int local_idx, Eigen::Vector3f& origin)>;

// 세션별 void map + keyframe 프레임 구축 (UFO 삽입은 모듈 내부)
SessionMaps buildVoidMap(int session_id, int map_size, const EvidenceParams& params,
                         float max_distance, const WorldScanLoader& load_world_scan);

std::unordered_set<int64_t> buildInterestSet(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                             float res);

void accumulateEvidence(const std::vector<FrameData>& frames,
                        const std::unordered_set<int64_t>& interest_set,
                        std::unordered_map<int64_t, VoidEvidence>& ev_map,
                        const EvidenceParams& params, float blind, float max_distance);

float computeVoidWeight(const VoidEvidence& ev, const EvidenceParams& params);

// ND/PD: nonground, UE: full map, StaticMap: full map compose
MapUpdateResult detectAndCompose(const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_nonground,
                                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_nonground,
                                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_full,
                                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_full,
                                 const SessionMaps& session1, const SessionMaps& session2,
                                 const EvidenceParams& params, float blind, float max_distance,
                                 float leaf_size);

}  // namespace lt_mapping
