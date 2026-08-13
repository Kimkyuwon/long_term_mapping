#include "voxel_evidence.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <ufo/map/ufomap.hpp>

#include "map_merge.hpp"
#include "session_context.hpp"

namespace lt_mapping {

using UfoVoidMap = ufo::Map<ufo::MapType::SEEN_FREE | ufo::MapType::REFLECTION>;

struct SessionMaps::Impl {
    UfoVoidMap map;
    std::vector<FrameData> frames;

    explicit Impl(float res)
        : map(static_cast<ufo::node_size_t>(res), static_cast<ufo::depth_t>(17))
    {
    }
};

SessionMaps::SessionMaps(float res) : impl_(std::make_unique<Impl>(res)) {}
SessionMaps::~SessionMaps() = default;
SessionMaps::SessionMaps(SessionMaps&&) noexcept = default;
SessionMaps& SessionMaps::operator=(SessionMaps&&) noexcept = default;

namespace {

ufo::PointCloud toUfoCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud)
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


void updateVoidEvidence(std::unordered_map<int64_t, VoidEvidence>& ev_map, int64_t key, float q_i,
                        const Eigen::Vector3f& u, int kf_id,
                        std::array<std::mutex, 64>& shard_mutex)
{
    std::size_t shard = static_cast<std::size_t>(key) & 63U;
    std::lock_guard<std::mutex> lock(shard_mutex[shard]);
    // 삽입 금지: 엔트리는 accumulateEvidence() 직렬 프리패스에서 이미 생성되어 있다.
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
                           std::unordered_map<int64_t, VoidEvidence>& ev_map,
                           const EvidenceParams& params, float blind, float max_distance,
                           std::array<std::mutex, 64>& shard_mutex)
{
    Eigen::Vector3f end(endpoint.x, endpoint.y, endpoint.z);
    Eigen::Vector3f ray = end - origin;
    float ray_len = ray.norm();
    if (ray_len <= std::max(params.res, 1e-6f) || ray_len <= blind) {
        return;
    }

    Eigen::Vector3f u = ray / ray_len;
    const float start_t = blind;
    const float end_t = std::min(ray_len, max_distance);
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
                float q_dist =
                    std::exp(-std::max(0.0f, r_i - params.r0) / std::max(params.r_s, 1e-3f));
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

}  // namespace

std::unordered_set<int64_t> buildInterestSet(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                                             float res)
{
    std::unordered_set<int64_t> keys;
    keys.reserve(cloud->points.size());
    for (const auto& p : cloud->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        keys.insert(voxelKey(p.x, p.y, p.z, res));
    }
    return keys;
}

void accumulateEvidence(const std::vector<FrameData>& frames,
                        const std::unordered_set<int64_t>& interest_set,
                        std::unordered_map<int64_t, VoidEvidence>& ev_map,
                        const EvidenceParams& params, float blind, float max_distance)
{
    ev_map.clear();
    ev_map.reserve(interest_set.size());
    for (int64_t key : interest_set) {
        ev_map.emplace(key, VoidEvidence{});
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
                                  interest_set, ev_map, params, blind, max_distance, shard_mutex);
        }
    }
    const double elapsed_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_begin).count();
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
                "[200.240 EvidenceAccum] frames=%zu interest_voxels=%zu dda_max_range_m=%.1f "
                "elapsed=%.2fs",
                frames.size(), interest_set.size(), max_distance, elapsed_s);
}

float computeVoidWeight(const VoidEvidence& ev, const EvidenceParams& params)
{
    if (ev.q_sum <= 0.0f) {
        return params.w_min;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es;
    es.computeDirect(ev.M / std::max(ev.q_sum, 1e-6f), Eigen::EigenvaluesOnly);
    const auto& lam = es.eigenvalues();  // (0)=λ₃, (1)=λ₂, (2)=λ₁
    float lam1 = std::max(lam(2), 1e-9f);
    float lam2 = std::max(lam(1), 0.0f);
    float aniso = std::min(lam2 / lam1, 1.0f);
    float w_geom = std::clamp(aniso / std::max(params.aniso_typ, 1e-3f), 0.0f, 1.0f);
    float w_obs = std::min(1.0f, ev.q_sum / std::max(params.n_req, 1e-3f));
    return std::clamp(w_geom * w_obs, params.w_min, 1.0f);
}

ClassificationResult classifyChanges(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_nonground,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_nonground,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_full,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_full, const UfoVoidMap& map1,
    const UfoVoidMap& map2, const std::unordered_map<int64_t, VoidEvidence>& ev_1in2,
    const std::unordered_map<int64_t, VoidEvidence>& ev_2in1, const EvidenceParams& params)
{
    ClassificationResult result;

    std::unordered_map<int64_t, float> w_1in2;
    std::unordered_map<int64_t, float> w_2in1;
    w_1in2.reserve(ev_1in2.size());
    w_2in1.reserve(ev_2in1.size());

    for (const auto& kv : ev_1in2) {
        if (kv.second.q_sum <= 0.0f) {
            continue;
        }
        w_1in2.emplace(kv.first, computeVoidWeight(kv.second, params));
    }
    for (const auto& kv : ev_2in1) {
        if (kv.second.q_sum <= 0.0f) {
            continue;
        }
        w_2in1.emplace(kv.first, computeVoidWeight(kv.second, params));
    }

    auto buildRep = [&](const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
        std::unordered_map<int64_t, pcl::PointXYZI> rep;
        rep.reserve(cloud->size() / 4 + 1);
        for (const auto& p : cloud->points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                continue;
            }
            int64_t key = voxelKey(p.x, p.y, p.z, params.res);
            if (rep.find(key) == rep.end()) {
                rep.emplace(key, p);
            }
        }
        return rep;
    };

    // ND/PD: NonGround 대표점만
    const auto first_ng_rep = buildRep(first_nonground);
    const auto second_ng_rep = buildRep(second_nonground);
    // UE: Full 맵 대표점 (지면 포함 — 미탐험 영역 표현)
    const auto first_full_rep = buildRep(first_full);
    const auto second_full_rep = buildRep(second_full);

    std::unordered_map<int64_t, Persistence> pers_1;
    std::unordered_map<int64_t, Persistence> pers_2;
    pers_1.reserve(first_ng_rep.size());
    pers_2.reserve(second_ng_rep.size());

    constexpr int kHitCountCap = 10;
    auto voidEvidenceCount = [](const std::unordered_map<int64_t, VoidEvidence>& ev_map,
                                int64_t key) -> float {
        auto it = ev_map.find(key);
        if (it == ev_map.end()) {
            return 1.0f;
        }
        return std::max(1.0f, static_cast<float>(it->second.n_keyframes));
    };

    struct DilateProbe {
        int hit = 0;
        bool free_found = false;
    };
    auto dilateProbe = [&](const pcl::PointXYZI& p, const UfoVoidMap& other_map) -> DilateProbe {
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
                    const int h =
                        std::clamp(std::max(0, static_cast<int>(other_map.hits(q))), 0, kHitCountCap);
                    if (h > 0) {
                        out.hit = std::max(out.hit, h);
                    } else if (other_map.seenFree(q)) {
                        out.free_found = true;
                    }
                }
            }
        }
        return out;
    };

    // opposite 세션이 이 위치를 hit/free(또는 이웃 승계)로 관측했는지 — UE 판정용
    auto isObserved = [&](const pcl::PointXYZI& p, const UfoVoidMap& other_map) -> bool {
        ufo::Point q(p.x, p.y, p.z);
        if (other_map.hits(q) > 0 || other_map.seenFree(q)) {
            return true;
        }
        if (params.cross_dilate > 0) {
            const DilateProbe nb = dilateProbe(p, other_map);
            if (nb.hit > 0 || nb.free_found) {
                return true;
            }
        }
        return false;
    };

    // ── ND (nonground, pers_1) ────────────────────────────────────────────
    for (const auto& kv : first_ng_rep) {
        int64_t key = kv.first;
        const auto& p = kv.second;
        auto& ps = pers_1[key];
        const float l_prev = ps.log_odds;

        float w_v = params.w_min;
        auto wit = w_1in2.find(key);
        if (wit != w_1in2.end()) {
            w_v = wit->second;
        }

        ufo::Point q(p.x, p.y, p.z);
        bool is_void = map2.seenFree(q);
        int n_hit = std::clamp(std::max(0, static_cast<int>(map2.hits(q))), 0, kHitCountCap);

        float raw_l = l_prev;
        bool observed = false;

        if (n_hit > 0) {
            const float n_hit_eff = effectiveCount(static_cast<float>(n_hit), params.n_sat);
            raw_l += params.l_hit * n_hit_eff;
            if (ps.hit_cnt < std::numeric_limits<uint16_t>::max()) {
                ps.hit_cnt = static_cast<uint16_t>(
                    std::min<int>(std::numeric_limits<uint16_t>::max(), ps.hit_cnt + n_hit));
            }
            observed = true;
        }
        if (is_void) {
            const float n_void = voidEvidenceCount(ev_1in2, key);
            const float n_void_eff = effectiveCount(n_void, params.n_sat);
            raw_l -= w_v * params.l_void * n_void_eff;
            ps.void_w_sum += w_v;
            observed = true;
        }

        if (!observed && params.cross_dilate > 0) {
            const DilateProbe nb = dilateProbe(p, map2);
            if (nb.hit > 0) {
                const float n_hit_eff = effectiveCount(static_cast<float>(nb.hit), params.n_sat);
                raw_l += params.dilate_weight * params.l_hit * n_hit_eff;
                observed = true;
            } else if (nb.free_found) {
                const float n_void = voidEvidenceCount(ev_1in2, key);
                const float n_void_eff = effectiveCount(n_void, params.n_sat);
                raw_l -= params.dilate_weight * w_v * params.l_void * n_void_eff;
                observed = true;
            }
        }

        ps.log_odds = clampLogOdds(raw_l, params.l_max);
        (void)observed;
        if (sigmoid(ps.log_odds) < params.tau_del) {
            result.nd_voxels.insert(key);
        }
    }

    // ── PD (nonground, pers_2) ────────────────────────────────────────────
    for (const auto& kv : second_ng_rep) {
        int64_t key = kv.first;
        const auto& p = kv.second;
        auto& ps = pers_2[key];
        const float l_prev = ps.log_odds;

        float w_v = params.w_min;
        auto wit = w_2in1.find(key);
        if (wit != w_2in1.end()) {
            w_v = wit->second;
        }

        ufo::Point q(p.x, p.y, p.z);
        bool is_void = map1.seenFree(q);
        int n_hit = std::clamp(std::max(0, static_cast<int>(map1.hits(q))), 0, kHitCountCap);

        float raw_l = l_prev;
        bool observed = false;

        if (n_hit > 0) {
            const float n_hit_eff = effectiveCount(static_cast<float>(n_hit), params.n_sat);
            raw_l -= params.l_hit * n_hit_eff;
            if (ps.hit_cnt < std::numeric_limits<uint16_t>::max()) {
                ps.hit_cnt = static_cast<uint16_t>(
                    std::min<int>(std::numeric_limits<uint16_t>::max(), ps.hit_cnt + n_hit));
            }
            observed = true;
        }
        if (is_void) {
            const float n_void = voidEvidenceCount(ev_2in1, key);
            const float n_void_eff = effectiveCount(n_void, params.n_sat);
            raw_l += w_v * params.l_void * n_void_eff;
            ps.void_w_sum += w_v;
            observed = true;
        }

        if (!observed && params.cross_dilate > 0) {
            const DilateProbe nb = dilateProbe(p, map1);
            if (nb.hit > 0) {
                const float n_hit_eff = effectiveCount(static_cast<float>(nb.hit), params.n_sat);
                raw_l -= params.dilate_weight * params.l_hit * n_hit_eff;
                observed = true;
            } else if (nb.free_found) {
                const float n_void = voidEvidenceCount(ev_2in1, key);
                const float n_void_eff = effectiveCount(n_void, params.n_sat);
                raw_l += params.dilate_weight * w_v * params.l_void * n_void_eff;
                observed = true;
            }
        }

        ps.log_odds = clampLogOdds(raw_l, params.l_max);
        (void)observed;
        if (sigmoid(ps.log_odds) > params.tau_add) {
            result.pd_voxels.insert(key);
        }
    }

    // ── UE (full map) ─────────────────────────────────────────────────────
    for (const auto& kv : first_full_rep) {
        if (!isObserved(kv.second, map2)) {
            result.first_ue_voxels.insert(kv.first);
        }
    }
    for (const auto& kv : second_full_rep) {
        if (!isObserved(kv.second, map1)) {
            result.second_ue_voxels.insert(kv.first);
        }
    }

    for (const auto& p : first_nonground->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        int64_t key = voxelKey(p.x, p.y, p.z, params.res);
        if (result.nd_voxels.find(key) != result.nd_voxels.end()) {
            result.nd_cloud->points.push_back(p);
        }
    }
    for (const auto& p : second_nonground->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        int64_t key = voxelKey(p.x, p.y, p.z, params.res);
        if (result.pd_voxels.find(key) != result.pd_voxels.end()) {
            result.pd_cloud->points.push_back(p);
        }
    }
    for (const auto& p : first_full->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        int64_t key = voxelKey(p.x, p.y, p.z, params.res);
        if (result.first_ue_voxels.find(key) != result.first_ue_voxels.end()) {
            result.first_ue->points.push_back(p);
        }
    }
    for (const auto& p : second_full->points) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            continue;
        }
        int64_t key = voxelKey(p.x, p.y, p.z, params.res);
        if (result.second_ue_voxels.find(key) != result.second_ue_voxels.end()) {
            result.second_ue->points.push_back(p);
        }
    }

    return result;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr composeFinalMap(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_map,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_map, const ClassificationResult& cls,
    const UfoVoidMap& map1, const UfoVoidMap& map2, const EvidenceParams& params, float leaf_size,
    ComposeDiagnostics& diag)
{
    auto static_map(new pcl::PointCloud<pcl::PointXYZI>());
    static_map->reserve(first_map->points.size() + second_map->points.size());

    // 점 단위가 아니라 복셀 단위로 세어야 cls.*_ue_voxels.size() 와 같은 축으로 비교된다.
    std::unordered_set<int64_t> first_ue_dropped;
    std::unordered_set<int64_t> second_ue_dropped;

    for (const auto& p : first_map->points) {
        int64_t key = voxelKey(p.x, p.y, p.z, params.res);
        if (cls.nd_voxels.find(key) != cls.nd_voxels.end()) {
            continue;
        }
        if (map1.seenFree(ufo::Point(p.x, p.y, p.z))) {
            if (cls.first_ue_voxels.find(key) != cls.first_ue_voxels.end()) {
                first_ue_dropped.insert(key);
            }
            continue;
        }
        static_map->points.push_back(p);
    }

    for (const auto& p : second_map->points) {
        if (map2.seenFree(ufo::Point(p.x, p.y, p.z))) {
            const int64_t key = voxelKey(p.x, p.y, p.z, params.res);
            if (cls.second_ue_voxels.find(key) != cls.second_ue_voxels.end()) {
                second_ue_dropped.insert(key);
            }
            continue;
        }
        static_map->points.push_back(p);
    }

    diag.first_ue_dropped_by_seen_free = first_ue_dropped.size();
    diag.second_ue_dropped_by_seen_free = second_ue_dropped.size();

    pcl::VoxelGrid<pcl::PointXYZI> down_size_filter;
    down_size_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    down_size_filter.setInputCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr(static_map));

    auto final_map(new pcl::PointCloud<pcl::PointXYZI>());
    down_size_filter.filter(*final_map);
    final_map->width = final_map->points.size();
    final_map->height = 1;
    final_map->is_dense = false;
    return pcl::PointCloud<pcl::PointXYZI>::Ptr(final_map);
}

MapUpdateResult detectAndCompose(const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_nonground,
                                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_nonground,
                                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& first_full,
                                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& second_full,
                                 const SessionMaps& session1, const SessionMaps& session2,
                                 const EvidenceParams& params, float blind, float max_distance,
                                 float leaf_size)
{
    MapUpdateResult out;

    // interest / classify 는 비지면만. void map·ray 는 전체 스캔(session frames) 유지.
    auto is_1in2 = buildInterestSet(first_nonground, params.res);
    auto is_2in1 = buildInterestSet(second_nonground, params.res);

    std::unordered_map<int64_t, VoidEvidence> ev_1in2;
    std::unordered_map<int64_t, VoidEvidence> ev_2in1;
    accumulateEvidence(session2.impl().frames, is_1in2, ev_1in2, params, blind, max_distance);
    accumulateEvidence(session1.impl().frames, is_2in1, ev_2in1, params, blind, max_distance);

    out.cls = classifyChanges(first_nonground, second_nonground, first_full, second_full,
                              session1.impl().map, session2.impl().map, ev_1in2, ev_2in1, params);

    out.static_map = composeFinalMap(first_full, second_full, out.cls, session1.impl().map,
                                     session2.impl().map, params, leaf_size, out.compose_diag);

    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
                "[200.270 ChangeDetect] nd_nonground=%zu pd_nonground=%zu ue1_full=%zu ue2_full=%zu "
                "static=%zu",
                out.cls.nd_voxels.size(), out.cls.pd_voxels.size(), out.cls.first_ue_voxels.size(),
                out.cls.second_ue_voxels.size(), out.static_map->size());

    const auto dropRatio = [](size_t dropped, size_t total) -> double {
        return total == 0 ? 0.0 : static_cast<double>(dropped) / static_cast<double>(total);
    };
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
                "[400.440 ComposeFinal] ue1_total=%zu ue1_dropped_seenfree=%zu ue1_drop_ratio=%.4f "
                "ue2_total=%zu ue2_dropped_seenfree=%zu ue2_drop_ratio=%.4f",
                out.cls.first_ue_voxels.size(), out.compose_diag.first_ue_dropped_by_seen_free,
                dropRatio(out.compose_diag.first_ue_dropped_by_seen_free,
                          out.cls.first_ue_voxels.size()),
                out.cls.second_ue_voxels.size(), out.compose_diag.second_ue_dropped_by_seen_free,
                dropRatio(out.compose_diag.second_ue_dropped_by_seen_free,
                          out.cls.second_ue_voxels.size()));

    return out;
}


SessionMaps buildVoidMap(int session_id, int map_size, const EvidenceParams& params,
                         float max_distance, const WorldScanLoader& load_world_scan)
{
    SessionMaps session_map(params.res);
    session_map.impl().map.reserve(30'000'000);
    session_map.impl().frames.reserve(static_cast<size_t>(map_size));

    ufo::IntegrationParams integ;
    integ.min_range = 1.0f;
    integ.max_range = max_distance;
    integ.inflate_hits_dist = 0.2f;
    integ.inflate_unknown = 1;
    integ.ray_passthrough_hits = true;
    integ.parallel = true;

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < map_size; ++i) {
        Eigen::Vector3f origin;
        auto world_cloud = load_world_scan(i, origin);
        if (!world_cloud) {
            world_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        }

        FrameData frame;
        frame.kf_id = i;
        frame.origin = origin;
        frame.world_cloud = world_cloud;
        session_map.impl().frames.push_back(frame);

        ufo::PointCloud ucloud;
        ucloud.reserve(world_cloud->points.size());
        for (const auto& p : world_cloud->points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
                continue;
            }
            ucloud.emplace_back(ufo::Point(p.x, p.y, p.z));
        }
        ufo::insertPointCloud(session_map.impl().map, ucloud,
                              ufo::Point(origin.x(), origin.y(), origin.z()), integ, false);
    }
    session_map.impl().map.propagateModified();

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    RCLCPP_INFO(rclcpp::get_logger("LTmapping"),
                "[200.230 VoidMap] session=%d keyframes=%d elapsed=%.3fs", session_id,
                map_size, static_cast<double>(ms) / 1000.0);
    return session_map;
}

}  // namespace lt_mapping

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr PubMerge_map;

void initMergeMapPublisher(const std::shared_ptr<rclcpp::Node>& nh, const rclcpp::QoS& qos_viz)
{
    PubMerge_map = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/Merge_map", qos_viz);
}

void resetMergeMapPublisher()
{
    PubMerge_map.reset();
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
            // session_id가 1이든 2이든 getGlobalPose() 하나로 앵커 합성까지 포함해 처리한다
            // (세션1 앵커는 항등원이므로 결과는 기존과 동일).
            gtsam::Pose3 global_pose = getGlobalPose(session_id, local_idx);
            key_pose.x = global_pose.translation().x();
            key_pose.y = global_pose.translation().y();
            key_pose.z = global_pose.translation().z();
            key_pose.roll = global_pose.rotation().roll();
            key_pose.pitch = global_pose.rotation().pitch();
            key_pose.yaw = global_pose.rotation().yaw();

            Eigen::Matrix4f tf = createTransformMatrix(key_pose);
            origin = Eigen::Vector3f(tf(0, 3), tf(1, 3), tf(2, 3));

            const std::string scans_dir = (session_id == 1 ? sessions[0].scans_path : sessions[1].scans_path);
            auto cloud = loadRawScanPointCloud(scans_dir, local_idx);
            if (!cloud) {
                return pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
            }
            std::vector<int> nan_idx;
            pcl::removeNaNFromPointCloud(*cloud, *cloud, nan_idx);
            pcl::transformPointCloud(*cloud, *cloud, tf);
            return cloud;
        };
    };

    auto session1 = lt_mapping::buildVoidMap(1, sessions[0].size, persistence_params,
                                             static_cast<float>(MAX_DISTANCE), make_loader(1));
    auto session2 = lt_mapping::buildVoidMap(2, sessions[1].size, persistence_params,
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
