#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstdint>

namespace lt_mapping {

// 3D voxel index -> int64 키 (각 축 21bit, 부호 포함)
inline int64_t voxelKey(float x, float y, float z, float res) {
    int64_t ix = (int64_t)std::floor(x / res) & 0x1FFFFF;
    int64_t iy = (int64_t)std::floor(y / res) & 0x1FFFFF;
    int64_t iz = (int64_t)std::floor(z / res) & 0x1FFFFF;
    return ix | (iy << 21) | (iz << 42);
}

// void 증거 누적기 (관심 voxel에만 생성됨)
struct VoidEvidence {
    Eigen::Matrix3f M = Eigen::Matrix3f::Zero();  // Σ q_i · u_i u_iᵀ (방향 DOP용)
    float q_sum = 0.f;                             // Σ q_i (유효 관측량)
    uint16_t n_rays = 0;                           // 통과 ray 수
    uint16_t n_keyframes = 0;                      // 관측 keyframe 수
    int32_t last_kf = -1;
};

// 연속 keyframe 에서 같은 voxel 을 반복 관측한 것은 서로 강하게 상관되어 있어 독립 증거가 아니다.
// 선형 누적(n 회 관측 = 증거 n 배)은 로봇이 천천히 지나간 구역의 증거를 수십 배 과대평가한다.
// n_sat 부근에서 포화하는 체감 함수로 유효 독립 관측 수를 환산한다.
//   n -> 0 이면 n 에 접근, n -> inf 이면 n_sat 로 수렴.
inline float effectiveCount(float n, float n_sat)
{
    const float s = std::max(n_sat, 1e-3f);
    return s * (1.0f - std::exp(-std::max(n, 0.0f) / s));
}

// persistence 상태 (전 voxel 대상 아님 — ND/PD 후보 voxel만)
struct Persistence {
    float log_odds = 0.f;
    uint16_t hit_cnt = 0;
    float void_w_sum = 0.f;  // 가중 void 증거 합
};

struct EvidenceParams {
    float res = 0.2f;         // voxel 크기 [m]
    float r0 = 15.f;          // 거리 감쇠 시작 [m]
    float r_s = 20.f;         // 거리 감쇠 스케일 [m]
    float d_surf = 0.5f;      // 스침각 가중 적용 구간 [m]
    // 방향 다양성 기준: 정규화된 M의 lambda_2/lambda_1 이 이 값에 도달하면 w_geom=1.
    // 지상 로봇은 평면 경로를 주행하므로 lambda_3 는 항상 0 근처이고, 관측 가능한 평면 안에서의
    // 방향 확산만이 실제 정보량이다. 0.3 은 평면 내 +-51.7deg 부채꼴에 해당하는 잠정값이며,
    // 다음 실행의 "aniso histogram" 로그로 실측한 뒤 재조정할 것.
    float aniso_typ = 0.3f;
    // 아래 3개는 구 3D vDOP 지표 전용. 판정에는 더 이상 쓰이지 않고 진단 로그 비교용으로만 남는다.
    float eps_reg = 1e-3f;    // (legacy) M 정칙화
    float vdop_typ = 1.5f;    // (legacy) vDOP 정규화 기준
    float sigma_vdop = 1.0f;  // (legacy)
    float n_req = 3.f;        // 유효 관측 요구량
    float w_min = 0.05f;
    // 상관 관측 보정 포화 상수. effectiveCount() 의 상한이자 유효 독립 관측 수의 최대치다.
    // 이 값이 void/hit 증거의 최대 크기를 정하므로 판정 임계와 직결된다:
    //   ND 성립(순수 void) 조건은 w_v * l_void * n_sat > |logit(tau_del)| = 0.8473.
    //   n_sat=5, l_void=0.4 이면 w_v > 0.424 로, 비로소 w_v 가 판정을 지배한다.
    float n_sat = 5.f;
    // log-odds
    float l_hit = 0.85f;
    float l_void = 0.40f;
    // void 증거와 hit 증거를 같은 voxel 에서 동시에 누적(경합)하므로 두 항이 상쇄된 뒤에도
    // 합이 이 상한을 넘을 수 있다. 포화 이후의 증거를 잘라 한쪽 증거가 무한히 우세해지는 것을 막는다.
    float l_max = 5.f;
    float tau_del = 0.30f;    // p < tau_del -> 삭제(ND 확정)
    float tau_add = 0.70f;    // p > tau_add -> 생성(PD 확정)
    // 이웃 관용 교차 조회: 중심 voxel 에서 상대 세션의 상태를 전혀 못 찾았을 때만 반경
    // cross_dilate [voxel] 이웃까지 조회를 넓혀 상태를 승계한다. 0 이면 기능 비활성이고
    // 중심 조회만 수행하는 기존 동작과 완전히 동일하다.
    int cross_dilate = 1;
    // 이웃에서 승계한 증거의 감쇠 계수. 중심 관측이 아니라 위치 추론에 기반한 증거이므로
    // 중심 관측보다 반드시 약해야 한다. 0.5 는 중심 증거의 절반값이라는 잠정값이다.
    float dilate_weight = 0.5f;
    // [ND/PD 연결 성분 후처리]
    // 정합 오차로 생긴 "껍질"형 오검출은 상대 세션 표면에 달라붙은 낱개~소형 군집으로 나타나고,
    // 진짜로 사라지거나 새로 생긴 물체는 상대 세션이 비워둔 공간을 차지하는 덩어리로 나타난다.
    // 실측(ND 12301 / PD 11283 점): 낱개 ND 의 89% 가 2세션 표면에서 0.5 m 이내인 반면
    // 11 점 이상 군집은 중앙 이격이 0.44 m 이상으로 급증한다. 크기 기준만으로 분리가 된다.
    // 연결 반경. 입력 맵이 voxel_size(0.4 m) 로 다운샘플되어 있어 실제 점 간격(최근접 이웃 거리
    // 중앙값 0.27 m, p75 0.38 m)이 판정 격자(res=0.2)보다 크다. 0.35 를 쓰면 정상 이웃 연결의
    // 1/4 이상이 끊겨 실제 물체가 인위적으로 파편화되므로(최대 군집 654 -> 81 점) 0.5 가 적절하다.
    float cluster_eps = 0.5f;
    // 이 크기 미만의 연결 성분을 오검출로 보고 제거한다.
    // 1 이하이면 필터가 완전히 비활성화되어 후처리 이전 동작과 비트 단위로 동일하다(A/B 기준선).
    int min_cluster_size = 11;
};

}  // namespace lt_mapping
