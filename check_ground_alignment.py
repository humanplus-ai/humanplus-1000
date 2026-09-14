"""Compare body foot height against depth-backprojected ground (mocapworld Y-up)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .data_loader import build_video_to_depth_index, load_release_episode
from .geometry import depth_to_pointcloud, estimate_floor_y_from_body, transform_points

FOOT_INDICES = (7, 8, 10, 11)
FOOT_NAMES = {
    7: "left_ankle",
    8: "right_ankle",
    10: "left_foot",
    11: "right_foot",
}


@dataclass
class FrameGroundSample:
    """One frame of foot vs. depth-ground alignment."""

    motion_i: int
    video_i: int
    foot_y_min: float
    foot_y_p5: float
    foot_y_mean: float
    depth_ground_y_local: float
    depth_ground_y_global: float
    depth_points_near_feet: int
    foot_xz: tuple[float, float]


def _foot_stats(body: np.ndarray) -> tuple[float, float, float, tuple[float, float]]:
    """Return ``(min_y, p5_y, mean_y, foot_xz)``."""
    feet = body[list(FOOT_INDICES)]
    ys = feet[:, 1]
    finite = np.isfinite(ys)
    if not bool(np.any(finite)):
        return float("nan"), float("nan"), float("nan"), (float("nan"), float("nan"))
    ys = ys[finite]
    xz = feet[finite][:, [0, 2]].mean(axis=0)
    return (
        float(np.min(ys)),
        float(np.percentile(ys, 5.0)),
        float(np.mean(ys)),
        (float(xz[0]), float(xz[1])),
    )


def _estimate_depth_ground_y(
    depth_points_mocap: np.ndarray,
    foot_xz: tuple[float, float],
    *,
    local_radius_m: float = 0.75,
    global_percentile: float = 2.0,
    local_percentile: float = 5.0,
) -> tuple[float, float, int]:
    """Estimate ground Y from mocapworld depth points.

    Returns ``(local_ground_y, global_ground_y, near_feet_count)``.
    Local: low-percentile Y inside an XZ disk around the feet.
    Global: low-percentile Y of the full frame (often dominated by tabletops).
    """
    if depth_points_mocap.size == 0:
        return float("nan"), float("nan"), 0

    ys = depth_points_mocap[:, 1]
    finite = np.isfinite(ys)
    pts = depth_points_mocap[finite]
    if pts.shape[0] == 0:
        return float("nan"), float("nan"), 0

    global_y = float(np.percentile(pts[:, 1], global_percentile))

    cx, cz = foot_xz
    if not (np.isfinite(cx) and np.isfinite(cz)):
        return float("nan"), global_y, 0

    dx = pts[:, 0] - cx
    dz = pts[:, 2] - cz
    near = (dx * dx + dz * dz) <= float(local_radius_m) ** 2
    near_pts = pts[near]
    if near_pts.shape[0] < 32:
        return float("nan"), global_y, int(near_pts.shape[0])

    local_y = float(np.percentile(near_pts[:, 1], local_percentile))
    return local_y, global_y, int(near_pts.shape[0])


def collect_ground_samples(
    release_dir: str | Path,
    *,
    stride: int = 15,
    max_frames: int = -1,
    depth_downsample: int = 6,
    depth_max_points: int = 40000,
    near_plane: float = 0.45,
    far_plane: float = 4.0,
    local_radius_m: float = 0.75,
) -> tuple[list[FrameGroundSample], float]:
    """Load a session and sample foot-vs-ground metrics."""
    episode = load_release_episode(
        release_dir,
        load_depth=True,
        load_point_cloud=False,
    )
    if episode.depth is None:
        raise ValueError(f"{release_dir} has no depth/depth dataset")

    T_mocap_slam = episode.calib.T_mocapworld_slamworld.astype(np.float64)
    depth_lookup = build_video_to_depth_index(
        episode.depth_frame_indices, episode.num_video_frames
    )
    depth_min = float(episode.depth_attrs.get("min_depth_m", 0.1))
    depth_max = float(episode.depth_attrs.get("max_depth_m", 8.0))
    depth_vis_max = min(depth_max, far_plane)
    K = (
        episode.depth_intrinsics
        if episode.depth_intrinsics is not None
        else episode.calib.rectified_K
    )

    viewer_floor_y = estimate_floor_y_from_body(episode.body_keypoints)

    n = episode.num_motion_frames
    if max_frames > 0:
        n = min(n, max_frames)
    indices = list(range(0, n, max(1, stride)))

    samples: list[FrameGroundSample] = []
    for motion_i in indices:
        video_i = int(episode.sync_video_frame_indices[motion_i])
        if video_i < 0 or video_i >= episode.num_video_frames:
            continue
        depth_i = int(depth_lookup[video_i]) if video_i < len(depth_lookup) else -1
        if depth_i < 0:
            continue

        body = episode.body_keypoints[motion_i]
        foot_y_min, foot_y_p5, foot_y_mean, foot_xz = _foot_stats(body)

        depth_frame = episode.depth[depth_i]
        pts_cam, _ = depth_to_pointcloud(
            depth_frame,
            K,
            downsample_factor=depth_downsample,
            max_points=depth_max_points,
            near_plane=max(near_plane, depth_min),
            far_plane=depth_vis_max,
        )
        if pts_cam.shape[0] == 0:
            continue

        T_slam_cam = episode.slam_T_camera[video_i].astype(np.float64)
        T_mocap_cam = T_mocap_slam @ T_slam_cam
        pts_mocap = transform_points(T_mocap_cam, pts_cam)

        local_y, global_y, near_count = _estimate_depth_ground_y(
            pts_mocap,
            foot_xz,
            local_radius_m=local_radius_m,
        )
        samples.append(
            FrameGroundSample(
                motion_i=int(motion_i),
                video_i=int(video_i),
                foot_y_min=foot_y_min,
                foot_y_p5=foot_y_p5,
                foot_y_mean=foot_y_mean,
                depth_ground_y_local=local_y,
                depth_ground_y_global=global_y,
                depth_points_near_feet=near_count,
                foot_xz=foot_xz,
            )
        )

    return samples, viewer_floor_y


def _summarize_diff(name: str, diffs: np.ndarray) -> None:
    """Print a distribution summary for a set of differences."""
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        print(f"  {name}: (no valid samples)")
        return
    print(
        f"  {name}: n={diffs.size} | "
        f"p50={np.percentile(diffs, 50):+.4f} m | "
        f"p05={np.percentile(diffs, 5):+.4f} m | "
        f"p95={np.percentile(diffs, 95):+.4f} m | "
        f"mean={float(np.mean(diffs)):+.4f} m"
    )


def print_report(
    release_dir: str | Path,
    samples: list[FrameGroundSample],
    viewer_floor_y: float,
) -> None:
    """Print a human-readable alignment report."""
    print("=" * 72)
    print(f"Ground alignment check: {Path(release_dir).resolve()}")
    print(f"Valid sample frames: {len(samples)}")
    print(f"viewer feet-mode floor_y (estimate_floor_y_from_body): {viewer_floor_y:+.4f} m")
    print("viewer first_frame: first-frame sole (mesh min-y or foot joints)")
    print("viewer --ground-mode fixed: y = 0.0000 m (mocapworld)")
    print("-" * 72)

    if not samples:
        print("No usable samples (check depth / sync / slam).")
        return

    foot_min = np.asarray([s.foot_y_min for s in samples], dtype=np.float64)
    foot_p5 = np.asarray([s.foot_y_p5 for s in samples], dtype=np.float64)
    depth_local = np.asarray([s.depth_ground_y_local for s in samples], dtype=np.float64)
    depth_global = np.asarray([s.depth_ground_y_global for s in samples], dtype=np.float64)

    print("Body foot-joint Y (mocapworld):")
    print(
        f"  min  p50={np.percentile(foot_min, 50):+.4f} m | "
        f"p05={np.percentile(foot_min, 5):+.4f} m | "
        f"p95={np.percentile(foot_min, 95):+.4f} m"
    )
    print(
        f"  p5   p50={np.percentile(foot_p5, 50):+.4f} m | "
        f"p05={np.percentile(foot_p5, 5):+.4f} m | "
        f"p95={np.percentile(foot_p5, 95):+.4f} m"
    )

    valid_local = np.isfinite(depth_local)
    if bool(np.any(valid_local)):
        dl = depth_local[valid_local]
        print("Depth ground Y (0.75 m disk around feet, 5th percentile):")
        print(
            f"  p50={np.percentile(dl, 50):+.4f} m | "
            f"p05={np.percentile(dl, 5):+.4f} m | "
            f"p95={np.percentile(dl, 95):+.4f} m | "
            f"valid frames={int(valid_local.sum())}/{len(samples)}"
        )
    else:
        print("Depth ground Y (local): no valid samples (too few near-foot points)")

    dg = depth_global[np.isfinite(depth_global)]
    print("Depth full-frame low percentile Y (2nd; often tabletop-dominated):")
    print(
        f"  p50={np.percentile(dg, 50):+.4f} m | "
        f"p05={np.percentile(dg, 5):+.4f} m | "
        f"p95={np.percentile(dg, 95):+.4f} m"
    )

    print("-" * 72)
    print("Foot - depth ground (positive = foot above depth ground):")

    local_mask = np.isfinite(depth_local)
    _summarize_diff("foot_y_min - depth_local", foot_min[local_mask] - depth_local[local_mask])
    _summarize_diff("foot_y_p5   - depth_local", foot_p5[local_mask] - depth_local[local_mask])
    _summarize_diff("foot_y_min - depth_global", foot_min - depth_global)
    _summarize_diff("viewer_floor_y - depth_local (const)", depth_local[local_mask] - viewer_floor_y)

    print("-" * 72)
    local_diff = foot_min[local_mask] - depth_local[local_mask]
    if local_diff.size == 0:
        print("Verdict: cannot decide (insufficient near-foot depth).")
        return

    p50 = float(np.percentile(local_diff, 50))
    if abs(p50) <= 0.05:
        verdict = "well aligned (|p50| <= 5 cm)"
    elif p50 > 0.05:
        verdict = f"feet above depth ground (floating, p50={p50:+.3f} m)"
    else:
        verdict = f"feet below depth ground (penetration, p50={p50:+.3f} m)"

    print(f"Verdict: {verdict}")
    print(
        "Prefer foot_y_min - depth_local; depth_global includes tabletops and "
        "should not be used as floor height by itself."
    )
    print("=" * 72)


def build_argparser() -> argparse.ArgumentParser:
    """CLI for the ground-alignment check."""
    parser = argparse.ArgumentParser(
        description="Compare body foot height with depth-backprojected ground (mocapworld)."
    )
    parser.add_argument(
        "--session-dir",
        "--release-dir",
        "--data-root",
        dest="session_dir",
        type=str,
        required=True,
        help="Session folder (contains annotation.hdf5)",
    )
    parser.add_argument("--stride", type=int, default=15, help="Sample stride (motion frames)")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max motion frames to inspect")
    parser.add_argument(
        "--local-radius-m",
        type=float,
        default=0.75,
        help="XZ radius (m) around the feet for local depth ground",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = build_argparser().parse_args(argv)
    samples, viewer_floor_y = collect_ground_samples(
        args.session_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        local_radius_m=args.local_radius_m,
    )
    print_report(args.session_dir, samples, viewer_floor_y)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
