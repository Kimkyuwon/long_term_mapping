# long_term_mapping

**Multi-Session LiDAR SLAM for Long-Term Map Maintenance**

> A ROS2 C++ package that merges independently-acquired LiDAR sessions, detects structural changes between sessions, and produces a unified map with positive/negative change maps. The implementation is inspired by the concepts in the [LT-mapper paper](#citation) and re-implemented from scratch for ROS2 Jazzy.

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
4. **Map change detection** — tile-based set-difference analysis identifying:
   - **ND (Negative Difference)**: points that disappeared between sessions
   - **PD (Positive Difference)**: newly appearing structures
5. **Curved Voxel Clustering** for dynamic object segmentation from the change maps
6. **Unified map export** — merged point cloud, per-session ground/non-ground maps, and updated pose files

```
Session 1 (directory1/)          Session 2 (directory2/)
  ├── Scans/  (*.pcd)               ├── Scans/  (*.pcd)
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
 ├── getLoopEdges()         — NanoGICP scan matching + DOP check
 ├── getPoses()             — add anchor-node prior/loop factors
 ├── runISAM2opt()          — iSAM2 pose-graph optimisation + updatePoses()
 ├── generateOptimizedMap() — assemble merged PCD maps
 ├── MapUpdate()            — tile-based ND/PD change detection
 └── saveEdges()            — write merged edge file
```

### Key Components

| Component | Role |
|---|---|
| **SOLiDModule** | Rotation-invariant global descriptor for inter-session loop candidates |
| **NanoGICP** | Fast generalised ICP for 6-DOF relative pose estimation |
| **GTSAM iSAM2** | Incremental Bayesian pose-graph optimiser |
| **DOP filter** | Dilution-of-Precision metric to reject geometrically degenerate loop/change detections |
| **CurvedVoxelClustering** | Curved-voxel-based Euclidean clustering for object-level segmentation |

---

## Input Data Format

Each session directory must follow this structure (compatible with the output of [pose_graph_optimization](../pose_graph_optimization)):

```
<session_dir>/
├── Scans/
│   ├── 0.pcd              ← full keyframe scan
│   ├── 0_ground.pcd       ← ground points only
│   ├── 0_nonground.pcd    ← non-ground points
│   ├── 1.pcd
│   └── ...
├── optimized_poses.txt    ← one line per keyframe:
│                              timestamp tx ty tz qx qy qz qw
└── edges.txt              ← one line per edge:
                               from_idx to_idx tx ty tz roll pitch yaw σ0…σ5
```

**`optimized_poses.txt` line format:**
```
<timestamp> <x> <y> <z> <qx> <qy> <qz> <qw>
```

**`edges.txt` line format:**
```
<from_idx> <to_idx> <tx> <ty> <tz> <roll> <pitch> <yaw> <σ0> <σ1> <σ2> <σ3> <σ4> <σ5>
```

---

## Dependencies

### ROS2 Packages
| Package | Version |
|---|---|
| ROS2 | Jazzy (Ubuntu 24.04) |
| `rclcpp` | ament |
| `pcl_ros` | ament |
| `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `visualization_msgs`, `tf2*` | standard |

### System Libraries
| Library | Purpose | Install |
|---|---|---|
| [PCL](https://pointclouds.org/) ≥ 1.12 | Point cloud processing | `sudo apt install libpcl-dev` |
| [Eigen3](https://eigen.tuxfamily.org/) ≥ 3.4 | Linear algebra | `sudo apt install libeigen3-dev` |
| [GTSAM](https://gtsam.org/) ≥ 4.1 | Pose-graph optimisation | see below |
| OpenMP | Parallelisation | `sudo apt install libomp-dev` |

### Third-party ROS2 Packages (source build)
| Package | Notes |
|---|---|
| [nano_gicp](https://github.com/engcang/nano_gicp) | Fast GICP implementation |
| [SOLiD](https://github.com/sparolab/solid) | Place recognition descriptor |
| [patchworkpp](https://github.com/url-kaist/patchwork-plusplus) | Ground segmentation (optional, commented out) |
| [fast_lio](https://github.com/Kimkyuwon/FAST_LIO_Localization_and_Mapping) | Upstream odometry / keyframe producer |

### GTSAM Installation
```bash
sudo add-apt-repository ppa:borglab/gtsam-release-4.1
sudo apt update
sudo apt install libgtsam-dev libgtsam-unstable-dev
```

---

## Build

```bash
# 1. Clone into workspace
cd ~/localization_ws/src
git clone https://github.com/Kimkyuwon/long_term_mapping.git

# 2. Install ROS2 dependencies
cd ~/localization_ws
rosdep install --from-paths src --ignore-src -r -y

# 3. Build
colcon build --packages-select long_term_mapping --cmake-args -DCMAKE_BUILD_TYPE=Release

# 4. Source workspace
source install/setup.bash
```

---

## Configuration

Edit `config/params.yaml` before launching:

```yaml
/**:
  ros__parameters:
    # ── Input session directories ──────────────────────────────────────────
    directory1: /path/to/session1          # Central (reference) session
    directory2: /path/to/session2          # Query (new) session
    output_directory: MergedOutput         # Relative to package root

    # ── Preprocessing ──────────────────────────────────────────────────────
    preprocess:
      blind: 1.0                           # Ignore returns within this radius [m]

    # ── SOLiD place recognition & pose-graph ───────────────────────────────
    posegraph:
      r_solid_thres: 0.95                  # SOLiD similarity threshold (↑ = stricter)
      fov_u:  15.0                         # LiDAR vertical FoV upper bound [deg]
      fov_d: -15.0                         # LiDAR vertical FoV lower bound [deg]
      num_angle: 120                       # SOLiD azimuth bins
      num_range: 100                       # SOLiD range bins
      num_height: 16                       # SOLiD height bins
      min_distance: 1                      # Minimum descriptor search range [m]
      max_distance: 100                    # Maximum descriptor search range [m]
      voxel_size: 0.4                      # Voxel leaf size for maps [m]
      num_exclude_recent: 0                # Exclude N most-recent frames from search
      num_candidates_from_tree: 20         # Top-K candidates per query
      dop_thres: 1.1                       # DOP ratio rejection threshold
```

### Parameter Tuning Guide

| Scenario | Recommendation |
|---|---|
| Dense urban, many loop candidates | Raise `r_solid_thres` (0.95–0.98) |
| Sparse environment / few revisits | Lower `r_solid_thres` (0.85–0.92) |
| 16-channel LiDAR | Set `num_height: 16`, adjust `fov_u/d` |
| 32/64-channel LiDAR | Set `num_height: 32` or `64` |
| Large voxel (fast, coarser map) | Raise `voxel_size` (0.5–1.0) |

---

## Running

### Launch with RViz

```bash
ros2 launch long_term_mapping lt_mapper.launch.py \
    config_path:=/path/to/config \
    config_file:=params.yaml \
    rviz:=true
```

### Launch without RViz

```bash
ros2 launch long_term_mapping lt_mapper.launch.py \
    rviz:=false
```

### Override session directories at launch time

```bash
ros2 run long_term_mapping LTmapping \
    --ros-args \
    -p directory1:=/data/session_A \
    -p directory2:=/data/session_B \
    -p output_directory:=Merged
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

| Topic | Type | Description |
|---|---|---|
| `/first_kf_node` | `sensor_msgs/PointCloud2` | Session 1 keyframe positions |
| `/second_kf_node` | `sensor_msgs/PointCloud2` | Session 2 keyframe positions |
| `/merge_kf_node` | `sensor_msgs/PointCloud2` | Merged keyframe positions |
| `/First_path` | `nav_msgs/Path` | Session 1 optimised trajectory |
| `/Second_path` | `nav_msgs/Path` | Session 2 optimised trajectory |
| `/Merge_path` | `nav_msgs/Path` | Combined trajectory |
| `/Merge_map` | `sensor_msgs/PointCloud2` | Full merged point cloud map |
| `/loopLine` | `visualization_msgs/Marker` | Inter-session loop constraint visualisation |
| `/lt_mapping_complete` | `std_msgs/Bool` | Published `true` upon completion |

---

## Output Files

All outputs are written to `<package_root>/<output_directory>/`:

```
<output_directory>/
├── FirstMap.pcd              ← Session 1 full map (optimised poses)
├── FirstGroundMap.pcd        ← Session 1 ground points
├── FirstNonGroundMap.pcd     ← Session 1 non-ground points
├── SecondMap.pcd             ← Session 2 full map
├── SecondGroundMap.pcd
├── SecondNonGroundMap.pcd
├── optimized_poses.txt       ← Merged pose list (same format as input)
├── edges.txt                 ← Merged edge list
├── Scans/                    ← Re-indexed keyframe PCD files
│   ├── 0.pcd, 0_ground.pcd, 0_nonground.pcd
│   └── ...
└── Debug/
    ├── ND.pcd                ← Negative-difference (disappeared) points
    ├── PD.pcd                ← Positive-difference (appeared) points
    ├── FirstUE.pcd           ← Session 1 under-estimated change points
    └── SecondUE.pcd          ← Session 2 under-estimated change points
```

The `Debug/ND.pcd` and `Debug/PD.pcd` files can be fed into a downstream change-management module (e.g., static map construction, dynamic object removal).

---

## Algorithm Details

### 1. Inter-session Loop Detection (SOLiD)

SOLiD (Spatial Overlap with LiDAR Descriptor) builds a rotation-invariant 3D histogram from each keyframe scan parameterised in cylindrical coordinates `(angle, range, height)`. The cosine similarity between descriptors identifies candidate loop pairs across sessions without any initial alignment assumption.

### 2. Scan Matching with DOP Rejection

Candidate pairs are refined with NanoGICP. To reject geometrically degenerate matches (e.g., long corridors), a **Dilution of Precision (DOP)** metric is computed from the matched point distribution. The DOP ratio

```
DOP_ratio = matching_DOP / max(curr_DOP, target_DOP)
```

must fall below `dop_thres`; matches with a high DOP ratio indicate insufficient geometric constraint and are discarded.

### 3. Anchor-node Pose-graph Optimization

Following the anchor-node formulation [[Kim et al., 2010]](#citation), each session maintains its own coordinate frame. The central session anchor `Δ_C` is fixed with near-zero covariance; the query anchor `Δ_Q` has large initial covariance. Inter-session loop factors are expressed relative to both anchors, allowing iSAM2 to jointly estimate the sessions' internal drifts and their relative offset.

Robust Cauchy M-estimators are applied to all loop factors to handle false positives.

### 4. Tile-based Change Detection

After trajectory optimisation, the merged map is partitioned into 2D tiles. For each tile:
- Points from Session 1 not found within `voxel_size` in Session 2 → **ND candidates**
- Points from Session 2 not found within `voxel_size` in Session 1 → **PD candidates**

A secondary DOP check validates whether the local geometry has sufficient observability before accepting a candidate as a genuine change. High DOP-ratio tiles are flagged as under-estimated and stored separately (`*UE.pcd`).

### 5. Curved Voxel Clustering

`CurvedVoxelClustering` (CVC) segments the change maps into individual object-level clusters. Unlike standard Euclidean clustering, CVC operates in the original sensor coordinate and groups points that lie on the same curved surface, yielding more coherent object segments.

---

## Acknowledgements

This package integrates or adapts the following open-source works:

| Library / Code | Authors | License | Link |
|---|---|---|---|
| **SOLiD descriptor** | Hyungtae Lim et al. | MIT | [sparolab/solid](https://github.com/sparolab/solid) |
| **NanoGICP** | Ken Nakamura | MIT | [engcang/nano_gicp](https://github.com/engcang/nano_gicp) |
| **nanoflann** | Jose Luis Blanco | BSD-2 | [jlblancoc/nanoflann](https://github.com/jlblancoc/nanoflann) |
| **GTSAM** | Frank Dellaert et al. | BSD-2 | [borglab/gtsam](https://github.com/borglab/gtsam) |
| **CurvedVoxelClustering** | btran | MIT | based on [Curved-Voxel-Clustering](https://github.com/wangx1996/Curved-Voxel-Clustering) |

---

## Citation

This package is a ROS2 re-implementation inspired by the multi-session SLAM and change detection concepts described in:

```bibtex
@inproceedings{kim2021ltmapper,
  title     = {LT-mapper: A Modular Framework for LiDAR-based Lifelong Mapping},
  author    = {Kim, Giseop and Kim, Ayoung},
  booktitle = {arXiv preprint arXiv:2107.07712},
  year      = {2021}
}
```

> **Note:** This codebase was independently written for the ROS2 ecosystem and is **not** a fork or derivative of the original [LT-mapper repository](https://github.com/gisbi-kim/lt-mapper). The algorithm design references the above paper; all code is an original implementation by the author.

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

> Third-party components (SOLiD, NanoGICP, nanoflann, CurvedVoxelClustering) retain their own licenses as listed in [Acknowledgements](#acknowledgements). All third-party licenses (MIT / BSD-2) are compatible with BSD-2-Clause.
