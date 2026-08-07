# long_term_mapping

**Multi-Session LiDAR SLAM for Long-Term Map Maintenance**

> A ROS2 C++ package that merges independently-acquired LiDAR sessions, detects structural changes between sessions, and produces a unified map with positive/negative change maps. The implementation is inspired by the concepts in the [LT-mapper paper](https://ieeexplore.ieee.org/abstract/document/9811916/) and re-implemented from scratch for ROS2 Jazzy.


| Before (single-session maps) | After (merged & change-detected) |
| ---------------------------- | -------------------------------- |
| before                       | after                            |


---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Input Data Format](#input-data-format)
- [Dependencies](#dependencies)
- [Build](#build)
- [Configuration](#configuration)
- [Running](#running)
- [Published Topics](#published-topics)
- [Output Files](#output-files)
- [Algorithm Details](#algorithm-details)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

---



## Overview

`long_term_mapping` takes two independently-recorded LiDAR sessions (each with its own pose-graph and keyframe scans) and performs:

1. **Inter-session place recognition** using [SOLiD](https://github.com/sparolab/solid) descriptors
2. **Multi-session pose-graph optimization** via GTSAM iSAM2 with anchor-node-based loop factors
3. **Geometric scan matching** using NanoGICP with DOP (Dilution of Precision)-based outlier rejection
4. **Void-evidence change detection** (UFOMap + log-odds) on non-ground structure points:
  - **ND (Negative Difference)**: non-ground structure that disappeared
  - **PD (Positive Difference)**: newly appearing non-ground structure
   - **UE (Unexplored)**: full-map voxels with no opposite-session void/hit evidence
5. **Unified map export** — `StaticMap.pcd`, debug ND/PD/UE clouds, and updated pose files

```
Session 1 (directory1/)          Session 2 (directory2/)
  ├── Scans/  (*.pcd)               ├── Scans/  (*.pcd)
  ├── StaticMap.pcd                 ├── StaticMap.pcd
  ├── optimized_poses.txt           ├── optimized_poses.txt
  └── edges.txt                     └── edges.txt
              │                              │
              └──────────┬───────────────────┘
                         ▼
              long_term_mapping node
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
     Merged map     ND / PD maps   Updated poses
```

---



## System Architecture

```
main()
 ├── setParams()            — load YAML parameters
├── getDirectory()         — resolve I/O paths
├── loadFiles()            — read poses & edges from both sessions
├── initNoises()           — initialise GTSAM noise models
├── getEdges()             — build intra-session odometry factors
├── placeRecognition()     — SOLiD-based inter-session loop detection
├── getLoopEdges()         — KISS-Matcher global registration + NanoGICP loop edges
├── getPoses()             — add anchor-node prior/loop factors
├── runISAM2opt()          — iSAM2 pose-graph optimization + updatePoses()
├── generateOptimizedMap() — assemble First/Second + Ground/NonGround maps
├── MapUpdate()            — void-evidence ND/PD/UE + StaticMap
 └── saveEdges()            — write merged edge file
```

`MapUpdate()` is a thin orchestrator in `src/long_term_mapping.cpp`. The change-detection
pipeline lives in `src/voxel_evidence.cpp` (`lt_mapping::buildVoidMap`,
`lt_mapping::detectAndCompose`).

### Key Components


| Component                        | Role                                                                         |
| -------------------------------- | ---------------------------------------------------------------------------- |
| **KISS-Matcher**                 | Global point cloud registration to compute initial inter-session transform   |
| **SOLiDModule**                  | Rotation-invariant global descriptor for inter-session loop candidates       |
| **NanoGICP**                     | Fast generalized ICP for 6-DOF relative pose estimation                      |
| **GTSAM iSAM2**                  | Incremental Bayesian pose-graph optimizer                                    |
| **DOP filter**                   | Dilution-of-Precision metric to reject geometrically degenerate loop matches |
| **UFOMap** (`thirdparty/ufomap`) | Per-session void/hit maps (`SEEN_FREE | REFLECTION`)                         |
| **voxel_evidence**               | Ray evidence, log-odds classify, StaticMap compose                           |


---



## Input Data Format

Each session directory must follow this structure (compatible with the output of [Pose_Graph_Optimization](https://github.com/Kimkyuwon/Pose_Graph_Optimization)):

```
<session_dir>/
├── Scans/
│   ├── 0.pcd              ← full keyframe scan
│   ├── 0_ground.pcd       ← ground points only
│   ├── 0_nonground.pcd    ← non-ground points
│   ├── 1.pcd
│   └── ...
├── StaticMap.pcd          ← merged static point cloud map (required for KISS-Matcher)
├── optimized_poses.txt    ← one line per keyframe:
│                              timestamp tx ty tz qx qy qz qw
└── edges.txt              ← one line per edge:
                               from_idx to_idx tx ty tz roll pitch yaw σ0…σ5
```

`optimized_poses.txt` **line format:**

```
<timestamp> <x> <y> <z> <qx> <qy> <qz> <qw>
```

`edges.txt` **line format:**

```
<from_idx> <to_idx> <tx> <ty> <tz> <roll> <pitch> <yaw> <σ0> <σ1> <σ2> <σ3> <σ4> <σ5>
```

---



## Dependencies



### System Libraries


| Library                                        | Purpose                                       | Install                          |
| ---------------------------------------------- | --------------------------------------------- | -------------------------------- |
| [PCL](https://pointclouds.org/) ≥ 1.12         | Point cloud processing                        | `sudo apt install libpcl-dev`    |
| [Eigen3](https://eigen.tuxfamily.org/) ≥ 3.4   | Linear algebra                                | `sudo apt install libeigen3-dev` |
| [GTSAM](https://gtsam.org/) ≥ 4.1              | Pose-graph optimization                       | see below                        |
| OpenMP                                         | Parallelization                               | `sudo apt install libomp-dev`    |
| [flann](https://github.com/flann-lib/flann)    | Approximate nearest neighbours (KISS-Matcher) | `sudo apt install libflann-dev`  |
| [lz4](https://github.com/lz4/lz4)              | Compression (KISS-Matcher)                    | `sudo apt install liblz4-dev`    |
| [oneTBB](https://github.com/oneapi-src/oneTBB) | Parallelism (KISS-Matcher)                    | `sudo apt install libtbb-dev`    |




### Bundled Third-party Libraries (no separate install)


| Library                                                   | Purpose                                       | Notes                                     |
| --------------------------------------------------------- | --------------------------------------------- | ----------------------------------------- |
| [KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher) | Global point cloud registration               | Source bundled in `include/kiss_matcher/` |
| [UFOMap](https://github.com/UnknownFreeOccupied/ufomap)   | Void/hit volumetric maps for change detection | Source bundled in `thirdparty/ufomap/`    |




### Third-party ROS2 Packages (source build)


| Package                                                                                               | Notes                                                    |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| [nano_gicp](https://github.com/engcang/nano_gicp)                                                     | Fast GICP implementation                                 |
| [SOLiD](https://github.com/sparolab/solid)                                                            | Place recognition descriptor                             |
| [fast_lio2_mapping_and_localization](https://github.com/Kimkyuwon/fast_lio2_mapping_and_localization) | LiDAR-Inertial odometry / keyframe producer              |
| [Pose_Graph_Optimization](https://github.com/Kimkyuwon/Pose_Graph_Optimization)                       | Single-session pose-graph optimization & keyframe export |




### GTSAM Installation

```bash
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt update
sudo apt install libgtsam-dev libgtsam-unstable-dev
```

---



## Build



### 1. Install system dependencies

```bash
# Core build tools & SLAM libraries
sudo apt install libeigen3-dev libpcl-dev libomp-dev

# KISS-Matcher dependencies (bundled source, needs these system libs)
sudo apt install libflann-dev liblz4-dev libtbb-dev

# GTSAM
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt update
sudo apt install libgtsam-dev libgtsam-unstable-dev
```



### 2. Clone and build

```bash
# Clone into workspace
cd ~/your_workspace/src
git clone https://github.com/Kimkyuwon/long_term_mapping.git

# Install ROS2 package dependencies
cd ~/your_workspace
rosdep install --from-paths src --ignore-src -r -y

# Build
# Note: On the first build, CMake will automatically download ROBIN
#       (KISS-Matcher dependency) from the internet. Internet access required once.
colcon build --packages-select long_term_mapping --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source workspace
source install/setup.bash
```

> **Note**: `include/kiss_matcher/` is bundled in this repository.  
> No separate KISS-Matcher installation is required.  
> ROBIN is fetched automatically by CMake on the first build and cached for subsequent builds.

---



## Configuration

Edit `config/params.yaml` before launching:

```yaml
/**:
  ros__parameters:
    # ── Input session directories ──────────────────────────────────────────
    directory1: /path/to/session1          # Central (reference) session
    directory2: /path/to/session2          # Query (new) session
    output_directory: output               # Relative to package root
    blind: 2.0                             # Ignore returns within this radius [m]

    # ── KISS-Matcher (inter-session global registration) ───────────────────
    anchor_resolution: 2.0                 # Voxel resolution for KISS-Matcher [m]

    # ── SOLiD place recognition & pose-graph ───────────────────────────────
    r_solid_thres: 0.95                    # SOLiD similarity threshold (↑ = stricter)
    fov_u:  22.5                           # LiDAR vertical FoV upper bound [deg]
    fov_d: -22.5                           # LiDAR vertical FoV lower bound [deg]
    num_angle: 120                         # SOLiD azimuth bins
    num_range: 50                          # SOLiD range bins
    num_height: 32                         # SOLiD height bins
    min_distance: 1                        # Minimum descriptor search range [m]
    max_distance: 50                       # Maximum descriptor search range [m]
    voxel_size: 0.4                        # Voxel leaf size for maps [m]
    num_exclude_recent: 0                  # Exclude N most-recent frames from search
    num_candidates_from_tree: 20           # Top-K candidates per query
    dop_thres: 1.3                         # DOP ratio rejection threshold

    # ── Void-evidence / persistence (change detection) ─────────────────────
    persistence.res: 0.2                   # Evidence / decision voxel size [m]
```

Other persistence knobs (`r0`, `r_s`, `d_surf`, `aniso_typ`, `n_req`, `w_min`, `n_sat`,
`l_hit`, `l_void`, `l_max`, `tau_del`, `tau_add`, `cross_dilate`, `dilate_weight`) live as
defaults in `EvidenceParams` (`include/voxel_evidence.hpp`). Only `persistence.res` is exposed
as a ROS parameter.

PD/ND are classified on `First/SecondNonGroundMap.pcd`. UE is classified on the full
`First/SecondMap.pcd`. Void maps still use full keyframe scans. `StaticMap` is composed from
the full maps so ground is retained (ND keys come only from non-ground).

#### Direction diversity

Void evidence accumulates `M = Σ qᵢ·uᵢuᵢᵀ`. Ground robots observe rays in a near-planar fan, so
`λ₃ ≈ 0` and 3D vDOP is uninformative. Direction diversity uses the in-plane spread:

```
aniso  = λ₂ / λ₁
w_geom = clamp(aniso / aniso_typ, 0, 1)
w_v    = clamp(w_geom * min(1, q_sum / n_req), w_min, 1)
```



#### Evidence competition and `n_sat`

`SEEN_FREE` and `REFLECTION` can both be true for one voxel; both contribute to the same log-odds:

```
# pers_1 — session-1 structure still present
if (n_hit  > 0)  log_odds += l_hit  * effectiveCount(n_hit,  n_sat)
if (is_void)     log_odds -= w_v * l_void * effectiveCount(n_void, n_sat)

# pers_2 — session-2 structure is new  (sign mirror)
if (n_hit  > 0)  log_odds -= l_hit  * effectiveCount(n_hit,  n_sat)
if (is_void)     log_odds += w_v * l_void * effectiveCount(n_void, n_sat)

log_odds = clamp(log_odds, ±l_max)
UE  <=>  no center evidence and no neighbor-inherited evidence
```

```
effectiveCount(n, n_sat) = n_sat · (1 − exp(−n / n_sat))
```



#### Neighbor-tolerant cross query

When the center cell has neither hit nor free evidence and `cross_dilate > 0`, the query widens to
the neighborhood. Hit evidence outranks free; inherited terms are scaled by `dilate_weight`.
`cross_dilate: 0` disables this (center-only). UE voxels are never dropped from the final map.

---



## Running



### Launch

```bash
ros2 launch long_term_mapping lt_mapper.launch.py 
```



### Expected Console Output

```
=== Parameters loaded ===
directory1: /data/Campus1
directory2: /data/Campus2
output_directory: Merged
[LTmapping] Session Edge Loading Complete.
[LTmapping] Place Recognition Complete.
[LTmapping] Loop Edge Generation Complete. size : 42
[LTmapping] Pose Factor loading Complete.
[LTmapping] Graph Optimization Complete.
[LTmapping] Map Merging Complete.
[LTmapping] Map Update Complete.
[LTmapping] Long Term SLAM Complete.
[LTmapping] Completion message published.
```

---



## Published Topics


| Topic                  | Type                        | Description                                 |
| ---------------------- | --------------------------- | ------------------------------------------- |
| `/first_kf_node`       | `sensor_msgs/PointCloud2`   | Session 1 keyframe positions                |
| `/second_kf_node`      | `sensor_msgs/PointCloud2`   | Session 2 keyframe positions                |
| `/merge_kf_node`       | `sensor_msgs/PointCloud2`   | Merged keyframe positions                   |
| `/First_path`          | `nav_msgs/Path`             | Session 1 optimized trajectory              |
| `/Second_path`         | `nav_msgs/Path`             | Session 2 optimized trajectory              |
| `/Merge_path`          | `nav_msgs/Path`             | Combined trajectory                         |
| `/Merge_map`           | `sensor_msgs/PointCloud2`   | Full merged point cloud map                 |
| `/loopLine`            | `visualization_msgs/Marker` | Inter-session loop constraint visualization |
| `/lt_mapping_complete` | `std_msgs/Bool`             | Published `true` upon completion            |


---



## Output Files

All outputs are written to `<package_root>/<output_directory>/`:

```
<output_directory>/
├── FirstMap.pcd              ← Session 1 full map (optimized poses)
├── FirstGroundMap.pcd        ← Session 1 ground points
├── FirstNonGroundMap.pcd     ← Session 1 non-ground points (PD/ND input)
├── SecondMap.pcd             ← Session 2 full map
├── SecondGroundMap.pcd
├── SecondNonGroundMap.pcd
├── StaticMap.pcd             ← Composed static map (ND + self-void removed)
├── optimized_poses.txt       ← Merged pose list (same format as input)
├── edges.txt                 ← Merged edge list
├── Scans/                    ← Re-indexed keyframe PCD files
│   ├── 0.pcd, 0_ground.pcd, 0_nonground.pcd
│   └── ...
└── Debug/
    ├── KissMatchedMap.pcd    ← Session 1 + 2 maps aligned by KISS-Matcher (intensity-coded per session)
    ├── ND.pcd                ← Non-ground negative-difference (disappeared)
    ├── PD.pcd                ← Non-ground positive-difference (appeared)
    ├── FirstUE.pcd           ← Session 1 full-map unexplored (UE)
    └── SecondUE.pcd          ← Session 2 full-map unexplored (UE)
```

`Debug/ND.pcd` / `PD.pcd` are structure-change clouds (non-ground only). UE clouds use the full
map. `StaticMap.pcd` is published on `/Merge_map`.

---



## Algorithm Details



### 1. Global Alignment & Anchor-node Pose-graph Optimization



#### 1-1. Why global alignment first?

Each session builds its own local map from an arbitrary starting pose, so their coordinate frames are unrelated. GTSAM's iSAM2 is a **nonlinear** optimizer that linearizes the cost function around the current estimate at every iteration. If Session 2 nodes are initialized far from their true positions in Session 1's frame, two failure modes occur:

1. **Linearization error** — the Jacobian computed at a wrong operating point points in the wrong direction, causing divergence or convergence to a bad local minimum.
2. **Cauchy kernel saturation** — loop closure edges use a Cauchy robust kernel (parameter = 1.0). When the residual of a loop edge greatly exceeds the Cauchy threshold, the kernel saturates and the edge's effective weight drops to near zero. Even with many correct inter-session loops, the optimizer treats them all as outliers and ignores them.



#### 1-2. Initial inter-session transform via KISS-Matcher

Before any keyframe-level loop detection, **KISS-Matcher** registers the two full static maps to compute a 6-DOF rigid body transform that maps Session 2's coordinate frame into Session 1's world frame, providing the initial estimate for all Session 2 nodes and placing them close enough to their true positions for iSAM2 to converge reliably.


| Parameter           | Description                                                                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `anchor_resolution` | Voxel downsampling resolution for KISS-Matcher [m]. Larger values are faster but less accurate. Typical range: 1.0–3.0 m |


The two maps, aligned by the resulting transform and colour-coded by session (intensity `1` = Session 1, `2` = Session 2), are saved to `Debug/KissMatchedMap.pcd` so the global registration quality can be inspected before pose-graph optimization runs.

> `StaticMap.pcd` must be the static-only point cloud map (dynamic objects removed), compatible with the output of [Pose_Graph_Optimization](https://github.com/Kimkyuwon/Pose_Graph_Optimization).



#### 1-3. Anchor-node pose-graph optimization

Following the anchor-node formulation [[Kim et al.]](https://ieeexplore.ieee.org/abstract/document/9811916/), Session 1's first node is fixed with near-zero covariance prior (`Δ_C`), while Session 2's first node is given a large covariance prior (`Δ_Q`), allowing it to be corrected by inter-session loop factors. iSAM2 then jointly optimizes the intra-session drifts of both sessions and their inter-session alignment. 

### 2. Inter-session Loop Detection (SOLiD)

SOLiD (Spatial Overlap with LiDAR Descriptor) builds a rotation-invariant 3D histogram from each keyframe scan parameterized in cylindrical coordinates `(angle, range, height)`. The cosine similarity between descriptors identifies candidate loop pairs across sessions without any initial alignment assumption.

### 3. Scan Matching with DOP Rejection

Candidate pairs are refined with NanoGICP. To reject geometrically degenerate matches (e.g., long corridors), a **Dilution of Precision (DOP)** metric is computed from the matched point distribution.

`computeDOP()` first voxelizes the point cloud (leaf size `DOP_VOXEL_SIZE = 2.5 m`) and computes the raw PDOP from the unit line-of-sight vectors between each remaining point and the query position, mirroring GNSS-style geometric dilution of precision. This raw value is then normalized into a score `rho`:

```
rho = pdop * sqrt(N_typical) / g_floor
```

where `N_typical` is the expected point count of a well-constrained scan derived from the configured vertical FoV (`fov_u`/`fov_d`) and a nominal sensing range (`MEAN_RANGE = 10 m`), and `g_floor` is a FoV-dependent geometric floor factor. Normalizing this way keeps the score comparable across sessions with different point densities or sensor FoVs.

The DOP ratio

```
DOP_ratio = matching_DOP / max(curr_DOP, target_DOP)
```

must fall below `dop_thres`, and the normalized `matching_DOP` itself must stay below an absolute cap (`1.2`); matches failing either check indicate insufficient geometric constraint and are discarded.

### 4. Void-evidence Change Detection

Implemented in `src/voxel_evidence.cpp`. Overview of the data flow:

```
full keyframe scans ──► UFOMap void/hit maps (SEEN_FREE | REFLECTION)
NonGround maps      ──► interest + log-odds classify ──► ND / PD
full maps           ──► opposite-session observe check ──► UE
full maps           ──► compose StaticMap (drop ND + self-seenFree; keep UE)
```

1. `buildVoidMap` — insert each session's full keyframe scans into a UFOMap
  (`SEEN_FREE | REFLECTION`) with `inflate_unknown=1`, then `propagateModified()`.
2. `detectAndCompose` — interest/evidence/ND/PD from `First/SecondNonGroundMap.pcd`;
  UE from full `First/SecondMap.pcd` (no opposite-session hit/free, with optional dilate).
3. `composeFinalMap` — start from full maps, remove session-1 ND voxels and
  self-`seenFree` dynamic points, keep session-2 points conservatively, voxel-downsample.

Ground is excluded from PD/ND (registration shells) but kept in UE so unexplored ground
regions remain visible. See [Configuration](#configuration) for `persistence.*` knobs.

---



## Related Projects

- [FAST-LIO based LiDAR SLAM & Localizaiton](https://github.com/Kimkyuwon/fast_lio2_mapping_and_localization): LiDAR-Inertial SLAM package with DOP-based scan matching confidence evaluation, supporting both Mapping and Localization modes
- [Pose-Graph-Optimization](https://github.com/Kimkyuwon/Pose_Graph_Optimization): LiDAR-based pose graph optimization backend with loop closure detection and dynamic object removal
- [SLAM-WebGUI](https://github.com/Kimkyuwon/ROS-SLAM-WebUI): Web-based GUI integrating SLAM, localization, and real-time visualization into a single browser interface — fully compatible with this package for convenient browser-based control

---



## Acknowledgements

This package integrates or adapts the following open-source works:


| Library / Code       | Authors                        | License | Link                                                                        |
| -------------------- | ------------------------------ | ------- | --------------------------------------------------------------------------- |
| **KISS-Matcher**     | Hyungtae Lim et al.            | MIT     | [MIT-SPARK/KISS-Matcher](https://github.com/MIT-SPARK/KISS-Matcher)         |
| **ROBIN**            | MIT-SPARK Lab                  | MIT     | [MIT-SPARK/ROBIN](https://github.com/MIT-SPARK/ROBIN)                       |
| **SOLiD descriptor** | Hogyun Kim et al.              | MIT     | [sparolab/solid](https://github.com/sparolab/solid)                         |
| **NanoGICP**         | Ken Nakamura                   | MIT     | [engcang/nano_gicp](https://github.com/engcang/nano_gicp)                   |
| **GTSAM**            | Frank Dellaert et al.          | BSD-2   | [borglab/gtsam](https://github.com/borglab/gtsam)                           |
| **UFOMap**           | Daniel Duberg, Patric Jensfelt | BSD-3   | [UnknownFreeOccupied/ufomap](https://github.com/UnknownFreeOccupied/ufomap) |


---



## License

```
BSD 2-Clause License

Copyright (c) 2025, Kyuwon Kim
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

> Third-party components (SOLiD, NanoGICP, nanoflann, UFOMap, KISS-Matcher) retain their own licenses as listed in [Acknowledgements](#acknowledgements). MIT / BSD-2 / BSD-3 are compatible with BSD-2-Clause.

