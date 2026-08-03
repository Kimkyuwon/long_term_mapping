#!/usr/bin/env python3
"""Spatial cluster analysis of long_term_mapping change-detection outputs.

Read-only: parses the ND / PD / UE / evidence PCDs produced by the ROS2 node and
asks whether the change decisions correspond to physically coherent objects
(compact blobs) or to registration noise (isolated voxels smeared over surfaces).

Usage:
    python3 analyze_changes.py [--debug-dir DIR] [--save-dir DIR] [--eps 0.35 0.71]
"""

import argparse
import os
import re
import struct
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SAVE_DIR = os.path.join(PKG_ROOT, "\u314a\u314a\u314a")
DEFAULT_DEBUG_DIR = os.path.join(DEFAULT_SAVE_DIR, "Debug")

RES = 0.2          # persistence.res : decision grid
VOXEL_SIZE = 0.4   # map downsample leaf size

SIZE_BINS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 30),
             (31, 100), (101, 300), (301, 1000), (1001, 10 ** 12)]
SIZE_LABELS = ["1", "2", "3-5", "6-10", "11-30",
               "31-100", "101-300", "301-1000", "1000+"]


# --------------------------------------------------------------------------- IO
def _lzf_decompress(src, expected):
    """Minimal LZF decoder (only needed for DATA binary_compressed)."""
    dst = bytearray(expected)
    si = di = 0
    n = len(src)
    while si < n:
        ctrl = src[si]
        si += 1
        if ctrl < 32:
            cnt = ctrl + 1
            dst[di:di + cnt] = src[si:si + cnt]
            si += cnt
            di += cnt
        else:
            length = ctrl >> 5
            if length == 7:
                length += src[si]
                si += 1
            ref = di - ((ctrl & 0x1F) << 8) - src[si] - 1
            si += 1
            for _ in range(length + 2):
                dst[di] = dst[ref]
                di += 1
                ref += 1
    return bytes(dst)


def read_pcd(path):
    """Return (N,4) float64 array [x, y, z, intensity] from a PCL PCD file."""
    with open(path, "rb") as fh:
        raw = fh.read()

    hdr_end = raw.find(b"DATA")
    line_end = raw.find(b"\n", hdr_end)
    header = raw[:line_end].decode("ascii", errors="replace")
    body = raw[line_end + 1:]

    def field(name):
        m = re.search(r"^%s\s+(.*)$" % name, header, re.MULTILINE)
        return m.group(1).strip() if m else ""

    fields = field("FIELDS").split()
    sizes = [int(v) for v in field("SIZE").split()]
    types = field("TYPE").split()
    counts = [int(v) for v in field("COUNT").split()] or [1] * len(fields)
    npts = int(field("POINTS") or 0)
    data_kind = field("DATA").split()[0]

    np_type = {("F", 4): "f4", ("F", 8): "f8", ("U", 1): "u1", ("U", 2): "u2",
               ("U", 4): "u4", ("I", 1): "i1", ("I", 2): "i2", ("I", 4): "i4"}
    dtype = np.dtype([(f, np_type[(t, s)], (c,) if c > 1 else ())
                      for f, s, t, c in zip(fields, sizes, types, counts)])

    if data_kind == "ascii":
        arr = np.loadtxt(path, skiprows=header.count("\n") + 1)
        arr = np.atleast_2d(arr)
        rec = {f: arr[:, i] for i, f in enumerate(fields)}
    else:
        if data_kind == "binary_compressed":
            comp_sz, uncomp_sz = struct.unpack("II", body[:8])
            blob = _lzf_decompress(body[8:8 + comp_sz], uncomp_sz)
            # compressed layout is field-major (SoA)
            rec, off = {}, 0
            for f, s, t, c in zip(fields, sizes, types, counts):
                nbytes = s * c * npts
                rec[f] = np.frombuffer(blob[off:off + nbytes],
                                       dtype=np_type[(t, s)], count=npts)
                off += nbytes
        else:
            data = np.frombuffer(body, dtype=dtype, count=npts)
            rec = {f: data[f] for f in fields}

    out = np.zeros((npts, 4), dtype=np.float64)
    for i, name in enumerate(("x", "y", "z")):
        out[:, i] = np.asarray(rec[name], dtype=np.float64).reshape(npts)
    if "intensity" in rec:
        out[:, 3] = np.asarray(rec["intensity"], dtype=np.float64).reshape(npts)
    return out


def voxel_key(pts, res=RES):
    """Replicate lt_mapping::voxelKey (21 bit per axis, floor division)."""
    idx = np.floor(pts[:, :3] / res).astype(np.int64) & 0x1FFFFF
    return idx[:, 0] | (idx[:, 1] << 21) | (idx[:, 2] << 42)


# -------------------------------------------------------------------- clustering
def cluster(pts, eps):
    """Connected components under an eps radius == DBSCAN(eps, min_samples=1)."""
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0
    tree = cKDTree(pts[:, :3])
    pairs = tree.query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n), n
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8),
                        (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    ncomp, labels = connected_components(graph, directed=False)
    return labels, ncomp


class Cluster(object):
    """One connected component with its shape descriptors."""

    __slots__ = ("size", "lo", "hi", "ctr", "ext", "agl")

    def __init__(self, blk, gm=None):
        self.size = len(blk)
        self.lo = blk.min(axis=0)
        self.hi = blk.max(axis=0)
        self.ctr = blk.mean(axis=0)
        if self.size >= 3:
            # sqrt of covariance eigenvalues == rms extent along principal axes.
            ev = np.linalg.eigvalsh(np.cov((blk - self.ctr).T))
            self.ext = np.sqrt(np.clip(ev, 0.0, None))  # ascending
        else:
            self.ext = np.zeros(3)
        self.agl = np.nan
        if gm is not None:
            a, ok = gm.height_above_ground(blk)
            if ok.any():
                self.agl = float(np.nanmean(a[ok]))

    @property
    def dims(self):
        return self.hi - self.lo

    def is_shell(self):
        """Axis-aligned test requested in the brief."""
        d = np.sort(self.dims)
        return bool(d[0] < 0.6 and d[1] >= 2.0 and d[2] >= 2.0)

    def is_planar(self):
        """PCA test: a flat sheet peeled off a wall/ground/roof surface.

        Independent of axis alignment, so it also catches slanted or curved-but-
        thin sheets that the bounding-box test misses.
        """
        return bool(self.size >= 20 and self.ext[0] < 0.20 and self.ext[2] >= 0.75)

    def fill(self):
        cells = np.prod(self.dims + RES) / RES ** 3
        return self.size / cells if cells > 0 else 1.0


def cluster_stats(pts, labels, gm=None):
    """Per-cluster descriptors, sorted by descending size."""
    order = np.argsort(labels, kind="stable")
    lab_s, pts_s = labels[order], pts[order]
    bounds = np.flatnonzero(np.diff(lab_s)) + 1
    starts = np.concatenate(([0], bounds))
    ends = np.concatenate((bounds, [len(lab_s)]))

    out = [Cluster(pts_s[s:e, :3], gm) for s, e in zip(starts, ends)]
    out.sort(key=lambda c: -c.size)
    sizes = np.array([c.size for c in out], dtype=np.int64)
    return out, sizes


def size_histogram(sizes):
    total = int(sizes.sum())
    rows = []
    for (lo, hi), label in zip(SIZE_BINS, SIZE_LABELS):
        sel = sizes[(sizes >= lo) & (sizes <= hi)]
        rows.append((label, len(sel), int(sel.sum()),
                     100.0 * sel.sum() / total if total else 0.0))
    return rows


# ------------------------------------------------------------------------ report
def hr(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def print_size_table(name, sizes, clusters):
    total = int(sizes.sum())
    print("\n[%s] clusters=%d  points=%d  largest=%d  mean=%.2f  median=%d"
          % (name, len(sizes), total, sizes.max() if len(sizes) else 0,
             sizes.mean() if len(sizes) else 0,
             int(np.median(sizes)) if len(sizes) else 0))
    print("  %-10s %10s %12s %10s" % ("size bin", "#clusters", "#points", "% points"))
    for label, nc, npts_, pct in size_histogram(sizes):
        print("  %-10s %10d %12d %9.2f%%" % (label, nc, npts_, pct))
    tiny = sizes[sizes <= 2]
    print("  --> singleton/pair (size 1-2): %d clusters, %d points = %.2f%% of all points"
          % (len(tiny), tiny.sum(), 100.0 * tiny.sum() / total if total else 0.0))
    big = sizes[sizes >= 31]
    print("  --> object-scale (size >=31) : %d clusters, %d points = %.2f%% of all points"
          % (len(big), big.sum(), 100.0 * big.sum() / total if total else 0.0))

    sh = [c for c in clusters if c.is_shell()]
    sh_pts = sum(c.size for c in sh)
    print("  --> bbox thin-shell (min axis <0.6 m, other two >=2 m): %d clusters, "
          "%d points = %.2f%% of all points"
          % (len(sh), sh_pts, 100.0 * sh_pts / total if total else 0.0))
    pl = [c for c in clusters if c.is_planar()]
    pl_pts = sum(c.size for c in pl)
    big_pts = sum(c.size for c in clusters if c.size >= 20)
    print("  --> PCA-planar sheets (>=20 pts, rms thickness <0.20 m, length >=0.75 m): "
          "%d clusters, %d points\n      = %.2f%% of all points, %.2f%% of points in "
          "clusters >=20 pts"
          % (len(pl), pl_pts, 100.0 * pl_pts / total if total else 0.0,
             100.0 * pl_pts / big_pts if big_pts else 0.0))


def print_top_table(name, clusters, topn=20):
    print("\n[%s] top %d clusters   (ext = rms extent on PCA axes; "
          "ext_min < 0.2 m => flat sheet)" % (name, topn))
    print("  %3s %7s %6s %6s %6s %8s %8s %8s %6s %6s %6s %6s %6s %5s"
          % ("#", "points", "dx", "dy", "dz", "cx", "cy", "cz",
             "agl", "e_min", "e_mid", "e_max", "fill", "flat"))
    for i, c in enumerate(clusters[:topn]):
        d = c.dims
        tag = "PLANE" if c.is_planar() else ("box" if c.is_shell() else "-")
        print("  %3d %7d %6.2f %6.2f %6.2f %8.2f %8.2f %8.2f %6.2f %6.2f %6.2f "
              "%6.2f %6.3f %5s"
              % (i + 1, c.size, d[0], d[1], d[2], c.ctr[0], c.ctr[1], c.ctr[2],
                 c.agl, c.ext[0], c.ext[1], c.ext[2], c.fill(), tag))


def z_histogram(name, z, step=0.5, label="z"):
    print("\n[%s] %s histogram (bin=%.1f m), n=%d" % (name, label, step, len(z)))
    if len(z) == 0:
        return
    lo = np.floor(z.min() / step) * step
    hi = np.ceil(z.max() / step) * step
    edges = np.arange(lo, hi + step, step)
    cnt, _ = np.histogram(z, bins=edges)
    for e, c in zip(edges[:-1], cnt):
        if c == 0:
            continue
        bar = "#" * int(round(60.0 * c / cnt.max()))
        print("  [%7.2f,%7.2f) %8d %6.2f%% %s"
              % (e, e + step, c, 100.0 * c / len(z), bar))


class GroundModel:
    """Per-cell minimum z of the reference map -> local ground height."""

    def __init__(self, ref_xyz, cell=1.0):
        self.cell = cell
        ij = np.floor(ref_xyz[:, :2] / cell).astype(np.int64)
        key = (ij[:, 0] + (1 << 20)) * (1 << 22) + (ij[:, 1] + (1 << 20))
        order = np.argsort(key, kind="stable")
        key_s, z_s = key[order], ref_xyz[order, 2]
        uniq, start = np.unique(key_s, return_index=True)
        self.keys = uniq
        self.zmin = np.minimum.reduceat(z_s, start)

    def height_above_ground(self, xyz):
        ij = np.floor(xyz[:, :2] / self.cell).astype(np.int64)
        key = (ij[:, 0] + (1 << 20)) * (1 << 22) + (ij[:, 1] + (1 << 20))
        pos = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
        ok = self.keys[pos] == key
        agl = np.full(len(xyz), np.nan)
        agl[ok] = xyz[ok, 2] - self.zmin[pos[ok]]
        return agl, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-dir", default=DEFAULT_DEBUG_DIR)
    ap.add_argument("--save-dir", default=DEFAULT_SAVE_DIR)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.35, 0.71])
    ap.add_argument("--primary-eps", type=float, default=None,
                    help="eps used for sections B-E (default: last of --eps)")
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    primary_eps = args.primary_eps if args.primary_eps is not None else args.eps[-1]
    out_dir = os.path.dirname(os.path.abspath(__file__))

    wanted = {
        "ND": os.path.join(args.debug_dir, "ND.pcd"),
        "PD": os.path.join(args.debug_dir, "PD.pcd"),
        "FirstUE": os.path.join(args.debug_dir, "FirstUE.pcd"),
        "SecondUE": os.path.join(args.debug_dir, "SecondUE.pcd"),
        "PersistenceMap": os.path.join(args.debug_dir, "PersistenceMap.pcd"),
        "EvidenceDebug": os.path.join(args.debug_dir, "EvidenceDebug.pcd"),
        "StaticMap": os.path.join(args.save_dir, "StaticMap.pcd"),
        "FirstMap": os.path.join(args.save_dir, "FirstMap.pcd"),
        "SecondMap": os.path.join(args.save_dir, "SecondMap.pcd"),
    }

    hr("0. INPUT FILES")
    clouds, missing = {}, []
    for name, path in wanted.items():
        if not os.path.isfile(path):
            missing.append(name)
            print("  MISSING  %-15s %s" % (name, path))
            continue
        clouds[name] = read_pcd(path)
        print("  ok       %-15s %8d pts  %s" % (name, len(clouds[name]), path))
    if missing:
        print("  -> missing files: %s (analysis continues with the rest)"
              % ", ".join(missing))

    # point spacing diagnostic: decides whether eps=0.35 is even admissible
    hr("0b. POINT SPACING DIAGNOSTIC (res=%.2f, map voxel=%.2f)" % (RES, VOXEL_SIZE))
    print("  %-15s %9s %9s %9s %9s %9s"
          % ("cloud", "nn_p05", "nn_p25", "nn_p50", "nn_p75", "nn_p95"))
    for name in ("ND", "PD", "FirstUE", "SecondUE"):
        if name not in clouds or len(clouds[name]) < 2:
            continue
        p = clouds[name][:, :3]
        d, _ = cKDTree(p).query(p, k=2)
        nn = d[:, 1]
        print("  %-15s %9.3f %9.3f %9.3f %9.3f %9.3f"
              % (name, *np.percentile(nn, [5, 25, 50, 75, 95])))
    if "StaticMap" in clouds:
        sm = clouds["StaticMap"][:, :3]
        sub = sm[np.random.default_rng(0).choice(len(sm), min(50000, len(sm)),
                                                 replace=False)]
        d, _ = cKDTree(sm).query(sub, k=2)
        print("  %-15s %9.3f %9.3f %9.3f %9.3f %9.3f"
              % ("StaticMap*", *np.percentile(d[:, 1], [5, 25, 50, 75, 95])))
    print("  (* sampled 50k). The map is voxel-downsampled at %.2f m, so the true "
          "neighbour\n     spacing -- not res=%.2f -- sets the admissible eps."
          % (VOXEL_SIZE, RES))

    # ground model is used both for per-cluster AGL (A) and the histograms (C)
    ref = clouds.get("StaticMap", clouds.get("PersistenceMap"))
    gm = GroundModel(ref[:, :3], cell=1.0) if ref is not None else None

    # ---------------------------------------------------------------- A
    results = {}
    for eps in args.eps:
        hr("A. CLUSTER ANALYSIS  (connected components, eps=%.2f m, min_samples=1)" % eps)
        for name in ("ND", "PD"):
            if name not in clouds:
                continue
            labels, _ = cluster(clouds[name], eps)
            cl, sizes = cluster_stats(clouds[name], labels, gm)
            results[(name, eps)] = (labels, cl, sizes)
            print_size_table("%s @ eps=%.2f" % (name, eps), sizes, cl)
            print_top_table("%s @ eps=%.2f" % (name, eps), cl, args.topn)

    # ------------------------------------------------------- A2 surface adhesion
    hr("A2. STAND-OFF FROM THE OTHER SESSION'S SURFACE vs CLUSTER SIZE (eps=%.2f)"
       % primary_eps)
    print("  A registration 'peel' is a duplicate of a surface the other session also")
    print("  observed, only shifted by the alignment error, so it sits within roughly")
    print("  one map spacing (~0.2-0.4 m) of that session's cloud. A genuinely removed")
    print("  or added object occupies space the other session left empty, so it must")
    print("  stand clearly further off.  ND is measured against the SECOND session map,")
    print("  PD against the FIRST -- never against StaticMap, which contains PD itself.")
    for name, ref_name in (("ND", "SecondMap"), ("PD", "FirstMap")):
        if (name, primary_eps) not in results or ref_name not in clouds:
            print("\n  [%s] skipped (need %s)" % (name, ref_name))
            continue
        labels, _, _ = results[(name, primary_eps)]
        d, _ = cKDTree(clouds[ref_name][:, :3]).query(clouds[name][:, :3], k=1)
        csize = np.bincount(labels)[labels]
        print("\n  [%s] distance to nearest %s point, by cluster size"
              % (name, ref_name))
        print("  %-14s %9s %9s %9s %9s %9s"
              % ("cluster size", "#points", "p25", "median", "p75", "%>0.5m"))
        for (lo, hi), label in zip(SIZE_BINS, SIZE_LABELS):
            m = (csize >= lo) & (csize <= hi)
            if m.sum() == 0:
                continue
            dd = d[m]
            print("  %-14s %9d %9.3f %9.3f %9.3f %8.1f%%"
                  % (label, m.sum(), np.percentile(dd, 25), np.median(dd),
                     np.percentile(dd, 75),
                     100.0 * np.count_nonzero(dd > 0.5) / len(dd)))
        tiny, big = d[csize <= 2], d[csize >= 31]
        if len(tiny) and len(big):
            print("  --> median stand-off  size 1-2: %.3f m   size >=31: %.3f m   "
                  "(ratio %.2fx)"
                  % (np.median(tiny), np.median(big),
                     np.median(big) / max(np.median(tiny), 1e-6)))

    # ---------------------------------------------------------------- B
    hr("B. ND <-> PD SPATIAL PAIRING (eps=%.2f, both sides restricted to >=5 pts)"
       % primary_eps)
    if ("ND", primary_eps) in results and ("PD", primary_eps) in results:
        _, nd_cl, _ = results[("ND", primary_eps)]
        _, pd_cl_all, _ = results[("PD", primary_eps)]
        pd_cl = [c for c in pd_cl_all if c.size >= 5]
        pd_ctr = np.array([c.ctr for c in pd_cl])
        pd_size = np.array([c.size for c in pd_cl])
        pd_tree = cKDTree(pd_ctr)

        cand = [c for c in nd_cl if c.size >= 5]
        print("  ND clusters >=5 pts: %d   PD clusters >=5 pts: %d"
              % (len(cand), len(pd_cl)))
        nd_ctr = np.array([c.ctr for c in cand])
        nd_size = np.array([c.size for c in cand])
        dist, idx = pd_tree.query(nd_ctr, k=1)

        for thr in (2.0, 5.0, 10.0):
            m = dist <= thr
            print("  ND clusters (>=5 pts, n=%d) with a PD cluster centroid within "
                  "%4.1f m: %5d (%.1f%%)  |  unpaired: %5d"
                  % (len(cand), thr, m.sum(), 100.0 * m.sum() / len(cand),
                     (~m).sum()))

        paired = dist <= 5.0
        if paired.any():
            ratio = np.minimum(nd_size[paired], pd_size[idx[paired]]) / \
                np.maximum(nd_size[paired], pd_size[idx[paired]])
            print("  size similarity of <5 m pairs: min/median/mean size-ratio "
                  "= %.3f / %.3f / %.3f  (1.0 = identical size)"
                  % (ratio.min(), np.median(ratio), ratio.mean()))
            print("  pairs with ratio >=0.5 (plausible same object moved): %d / %d (%.1f%%)"
                  % ((ratio >= 0.5).sum(), paired.sum(),
                     100.0 * (ratio >= 0.5).sum() / paired.sum()))

        print("\n  top %d ND clusters and their nearest PD cluster" % args.topn)
        print("  %3s %8s %9s %9s %9s %9s %9s"
              % ("#", "nd_pts", "cx", "cy", "cz", "dist_pd", "pd_pts"))
        for i in range(min(args.topn, len(cand))):
            c = cand[i]
            print("  %3d %8d %9.2f %9.2f %9.2f %9.2f %9d"
                  % (i + 1, c.size, c.ctr[0], c.ctr[1], c.ctr[2],
                     dist[i], pd_size[idx[i]]))

    # ---------------------------------------------------------------- C
    hr("C. HEIGHT DISTRIBUTION")
    if ref is not None:
        print("  reference map z min/median/max = %.2f / %.2f / %.2f  (span %.1f m,"
              "\n  so a single global ground level is meaningless -> also report height"
              "\n  above the local ground: per-1 m xy cell minimum z of the map)"
              % (ref[:, 2].min(), np.median(ref[:, 2]), ref[:, 2].max(),
                 ref[:, 2].max() - ref[:, 2].min()))
    for name in ("ND", "PD"):
        if name not in clouds:
            continue
        z_histogram(name, clouds[name][:, 2])
        if ref is None:
            continue
        agl, ok = gm.height_above_ground(clouds[name][:, :3])
        a = agl[ok]
        z_histogram(name, a, label="height-above-local-ground (AGL)")
        print("  cells resolved: %d/%d (%.1f%%)"
              % (ok.sum(), len(ok), 100.0 * ok.sum() / len(ok)))
        for lo, hi, tag in ((-1e9, 0.3, "ground shell (AGL <= 0.30 m)"),
                            (0.3, 2.0, "object band (0.3 .. 2.0 m)"),
                            (2.0, 1e9, "above 2.0 m (walls/roofs/canopy)")):
            m = (a > lo) & (a <= hi)
            print("  %-36s %8d (%.2f%%)" % (tag, m.sum(), 100.0 * m.sum() / len(a)))

    # ---------------------------------------------------------------- D
    hr("D. EVIDENCE WEIGHT w_v  vs  CLUSTER SIZE  (the decisive test)")
    if "EvidenceDebug" in clouds and ("ND", primary_eps) in results:
        ev = clouds["EvidenceDebug"]
        ev_key = voxel_key(ev)
        # keep max w_v per voxel key (they are unique in practice)
        order = np.argsort(ev_key, kind="stable")
        ev_key_s, ev_w_s = ev_key[order], ev[order, 3] / 100.0
        print("  EvidenceDebug: %d pts, %d unique voxel keys, w_v range %.4f..%.4f"
              % (len(ev), len(np.unique(ev_key)), ev_w_s.min(), ev_w_s.max()))

        for name, eps_used in [("ND", e) for e in args.eps]:
            print("\n  ---- %s clustered at eps=%.2f ----" % (name, eps_used))
            labels, cl, sizes = results[(name, eps_used)]
            pts = clouds[name]
            key = voxel_key(pts)
            pos = np.searchsorted(ev_key_s, key)
            pos = np.clip(pos, 0, len(ev_key_s) - 1)
            hit = ev_key_s[pos] == key
            w = np.where(hit, ev_w_s[pos], np.nan)
            print("  %s points matched to EvidenceDebug by voxel key (res=%.2f): "
                  "%d / %d (%.2f%%)"
                  % (name, RES, hit.sum(), len(pts), 100.0 * hit.sum() / len(pts)))

            csize = np.bincount(labels)[labels]
            print("\n  mean w_v grouped by the size of the cluster each point belongs to")
            print("  %-14s %9s %9s %9s %9s %9s %9s"
                  % ("cluster size", "#points", "mean w_v", "median", "p10", "p25",
                     "%w_v=1"))
            for (lo, hi), label in zip(SIZE_BINS, SIZE_LABELS):
                m = hit & (csize >= lo) & (csize <= hi)
                if m.sum() == 0:
                    continue
                ww = w[m]
                print("  %-14s %9d %9.4f %9.4f %9.4f %9.4f %8.1f%%"
                      % (label, m.sum(), ww.mean(), np.median(ww),
                         np.percentile(ww, 10), np.percentile(ww, 25),
                         100.0 * np.count_nonzero(ww >= 0.999) / len(ww)))

            tiny_m = hit & (csize <= 2)
            big_m = hit & (csize >= 31)
            if tiny_m.sum() and big_m.sum():
                a, b = w[tiny_m], w[big_m]
                print("\n  >>> singleton/pair clusters (1-2 pts) : n=%d  mean w_v=%.4f  "
                      "median=%.4f" % (len(a), a.mean(), np.median(a)))
                print("  >>> object-scale clusters (>=31 pts) : n=%d  mean w_v=%.4f  "
                      "median=%.4f" % (len(b), b.mean(), np.median(b)))
                diff = b.mean() - a.mean()
                pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
                print("  >>> difference (big - tiny) = %+.4f   Cohen's d = %+.3f"
                      % (diff, diff / pooled if pooled > 0 else 0.0))
                try:
                    from scipy.stats import mannwhitneyu
                    u, p = mannwhitneyu(b, a, alternative="two-sided")
                    print("  >>> Mann-Whitney U two-sided p = %.3e" % p)
                except Exception as exc:  # pragma: no cover
                    print("  (Mann-Whitney unavailable: %s)" % exc)
                corr = np.corrcoef(np.log10(csize[hit]), w[hit])[0, 1]
                print("  >>> Pearson corr( log10(cluster size), w_v ) over all matched "
                      "ND pts = %+.4f" % corr)

            # w_v of ND versus the full void2 candidate population
            nd_keyset = np.isin(ev_key_s, key)
            print("\n  w_v of all void2 candidates      : n=%d mean=%.4f median=%.4f"
                  % (len(ev_w_s), ev_w_s.mean(), np.median(ev_w_s)))
            print("  w_v of the subset promoted to ND : n=%d mean=%.4f median=%.4f"
                  % (nd_keyset.sum(), ev_w_s[nd_keyset].mean(),
                     np.median(ev_w_s[nd_keyset])))
            print("  w_v of candidates NOT promoted   : n=%d mean=%.4f median=%.4f"
                  % ((~nd_keyset).sum(), ev_w_s[~nd_keyset].mean(),
                     np.median(ev_w_s[~nd_keyset])))

    # ---------------------------------------------------------------- E
    hr("E. UNEXPLAINED (UE) RESIDUAL CLUSTERS (eps=%.2f)" % primary_eps)
    for name in ("FirstUE", "SecondUE"):
        if name not in clouds:
            continue
        labels, _ = cluster(clouds[name], primary_eps)
        cl, sizes = cluster_stats(clouds[name], labels, gm)
        results[(name, primary_eps)] = (labels, cl, sizes)
        print_size_table(name, sizes, cl)
        print_top_table(name, cl, 10)

    # ---------------------------------------------------------------- static map
    hr("F. StaticMap SUMMARY (no clustering)")
    if "StaticMap" in clouds:
        sm = clouds["StaticMap"]
        print("  points=%d" % len(sm))
        for i, ax in enumerate("xyz"):
            print("  %s: min=%9.2f  p25=%9.2f  median=%9.2f  p75=%9.2f  max=%9.2f"
                  % (ax, sm[:, i].min(), np.percentile(sm[:, i], 25),
                     np.median(sm[:, i]), np.percentile(sm[:, i], 75),
                     sm[:, i].max()))
        print("  unique voxels @res=%.2f : %d" % (RES, len(np.unique(voxel_key(sm)))))

    # ---------------------------------------------------------------- H
    hr("G. SYNTHESIS: per-point object-likeness (eps=%.2f)" % primary_eps)
    print("  Joint criterion. A point counts as object-like only if it is part of a")
    print("  cluster of >=11 points AND stands >0.35 m off the other session's cloud;")
    print("  a point is peel-like if its cluster has <=10 points AND it sits within")
    print("  0.35 m of that cloud. Anything else is ambiguous.")
    print("  %-6s %9s %14s %14s %13s" % ("cloud", "points", "object-like",
                                         "peel-like", "ambiguous"))
    for name, ref_name in (("ND", "SecondMap"), ("PD", "FirstMap")):
        if (name, primary_eps) not in results or ref_name not in clouds:
            continue
        labels, _, _ = results[(name, primary_eps)]
        d, _ = cKDTree(clouds[ref_name][:, :3]).query(clouds[name][:, :3], k=1)
        csize = np.bincount(labels)[labels]
        obj = (csize >= 11) & (d > 0.35)
        peel = (csize <= 10) & (d <= 0.35)
        amb = ~(obj | peel)
        n = len(d)
        print("  %-6s %9d %8d %5.1f%% %8d %5.1f%% %7d %5.1f%%"
              % (name, n, obj.sum(), 100.0 * obj.sum() / n,
                 peel.sum(), 100.0 * peel.sum() / n,
                 amb.sum(), 100.0 * amb.sum() / n))

    # ---------------------------------------------------------------- plots
    if not args.no_plot and "ND" in clouds and "PD" in clouds:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(20, 9))
            ax = axes[0]
            if "StaticMap" in clouds:
                sm = clouds["StaticMap"]
                s = sm[np.random.default_rng(1).choice(
                    len(sm), min(120000, len(sm)), replace=False)]
                ax.scatter(s[:, 0], s[:, 1], s=0.2, c="0.85", lw=0, label="StaticMap")
            ax.scatter(clouds["PD"][:, 0], clouds["PD"][:, 1], s=1.2, c="tab:blue",
                       lw=0, label="PD (new)")
            ax.scatter(clouds["ND"][:, 0], clouds["ND"][:, 1], s=1.2, c="tab:red",
                       lw=0, label="ND (disappeared)")
            ax.set_title("XY overview: ND (red) / PD (blue) over StaticMap")
            ax.set_aspect("equal")
            ax.legend(markerscale=8, loc="best")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

            ax = axes[1]
            if "StaticMap" in clouds:
                ax.scatter(s[:, 0], s[:, 1], s=0.2, c="0.9", lw=0)
            for key, colour, mark in (("ND", "tab:red", "o"), ("PD", "tab:blue", "s")):
                _, cl, _ = results[(key, primary_eps)]
                for rank, c in enumerate(cl[:args.topn]):
                    ax.add_patch(plt.Rectangle(
                        (c.lo[0], c.lo[1]), max(c.dims[0], 0.3),
                        max(c.dims[1], 0.3), fill=False, ec=colour, lw=1.4))
                    ax.annotate("%s%d:%d" % (key[0], rank + 1, c.size),
                                (c.ctr[0], c.ctr[1]), fontsize=6, color=colour)
            ax.set_title("Top-%d clusters (eps=%.2f): ND red / PD blue bounding boxes"
                         % (args.topn, primary_eps))
            ax.set_aspect("equal")
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")

            png1 = os.path.join(out_dir, "change_clusters_xy.png")
            fig.tight_layout()
            fig.savefig(png1, dpi=130)
            plt.close(fig)

            fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
            for ax, key, colour in ((axes[0], "ND", "tab:red"),
                                    (axes[1], "PD", "tab:blue")):
                _, _, sizes = results[(key, primary_eps)]
                ax.hist(sizes, bins=np.logspace(0, np.log10(max(sizes.max(), 10)), 40),
                        color=colour)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_title("%s cluster size distribution (eps=%.2f)"
                             % (key, primary_eps))
                ax.set_xlabel("points per cluster")
                ax.set_ylabel("#clusters")
            ax = axes[2]
            for key, colour in (("ND", "tab:red"), ("PD", "tab:blue")):
                ax.hist(clouds[key][:, 2], bins=np.arange(
                    np.floor(clouds[key][:, 2].min()),
                    np.ceil(clouds[key][:, 2].max()) + 0.5, 0.5),
                    histtype="step", color=colour, label=key, lw=1.6)
            ax.set_title("z distribution (0.5 m bins)")
            ax.set_xlabel("z [m]")
            ax.legend()
            png2 = os.path.join(out_dir, "change_size_height.png")
            fig.tight_layout()
            fig.savefig(png2, dpi=130)
            plt.close(fig)

            hr("H. FIGURES")
            print("  %s" % png1)
            print("  %s" % png2)
        except Exception as exc:
            print("  plotting failed: %s" % exc, file=sys.stderr)


if __name__ == "__main__":
    main()
