"""Geometry helpers: undistort, project, transform, skeleton, depth."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .constants import MANO21_PARENTS, SMPL24_PARENTS


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid ``4x4`` transform."""
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a ``4x4`` transform to ``(N, 3)`` points."""
    if points.size == 0:
        return points.reshape(0, 3).astype(np.float32)
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.concatenate([points.astype(np.float64), ones], axis=1)
    out = (T.astype(np.float64) @ homo.T).T
    return out[:, :3].astype(np.float32)


def transform_poses(T_target_source: np.ndarray, poses_source: np.ndarray) -> np.ndarray:
    """Transform a batch of ``(N, 4, 4)`` poses from source to target."""
    T = T_target_source.astype(np.float64)
    out = np.empty_like(poses_source, dtype=np.float64)
    for i in range(poses_source.shape[0]):
        out[i] = T @ poses_source[i].astype(np.float64)
    return out.astype(np.float32)


def estimate_floor_y_from_body(
    body_keypoints: np.ndarray,
    foot_indices: Tuple[int, ...] = (7, 8, 10, 11),
    sample_stride: int = 30,
    percentile: float = 5.0,
) -> float:
    """Estimate floor height ``y`` from mocapworld foot keypoints.

    Uses a low percentile rather than the median so stepping does not lift
    the estimated ground plane.
    """
    if body_keypoints.size == 0:
        return 0.0
    idx = np.arange(0, body_keypoints.shape[0], max(1, sample_stride))
    feet = body_keypoints[idx][:, list(foot_indices), 1].reshape(-1)
    feet = feet[np.isfinite(feet)]
    if feet.size == 0:
        return 0.0
    return float(np.percentile(feet, percentile))


def estimate_floor_y_first_frame(
    body_keypoints: np.ndarray,
    foot_indices: Tuple[int, ...] = (7, 8, 10, 11),
) -> float:
    """Lowest foot-joint ``y`` on the first valid frame (floor is static)."""
    if body_keypoints.size == 0:
        return 0.0
    for i in range(int(body_keypoints.shape[0])):
        feet = np.asarray(body_keypoints[i, list(foot_indices), 1], dtype=np.float64)
        finite = feet[np.isfinite(feet)]
        if finite.size > 0:
            return float(np.min(finite))
    return 0.0


def skeleton_lines(joints: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Build ``(E, 2, 3)`` bone segments from joints and parent indices."""
    lines = []
    for i, p in enumerate(parents.tolist()):
        if p < 0:
            continue
        lines.append([joints[p], joints[i]])
    if not lines:
        return np.zeros((0, 2, 3), dtype=np.float32)
    return np.asarray(lines, dtype=np.float32)


def smpl24_skeleton_lines(joints: np.ndarray) -> np.ndarray:
    """SMPL-24 skeleton line segments."""
    return skeleton_lines(joints, SMPL24_PARENTS)


def mano21_skeleton_lines(joints: np.ndarray) -> np.ndarray:
    """MANO-21 skeleton line segments."""
    return skeleton_lines(joints, MANO21_PARENTS)


def make_fisheye_undistort_maps(
    K: np.ndarray,
    D: np.ndarray,
    src_size_wh: Tuple[int, int],
    new_K: np.ndarray,
    dst_size_wh: Tuple[int, int],
    R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build OpenCV fisheye undistort/rectify maps."""
    del src_size_wh
    R_use = np.eye(3, dtype=np.float64) if R is None else np.asarray(R, dtype=np.float64)
    D4 = np.asarray(D, dtype=np.float64).reshape(4, 1)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        np.asarray(K, dtype=np.float64),
        D4,
        R_use,
        np.asarray(new_K, dtype=np.float64),
        dst_size_wh,
        cv2.CV_16SC2,
    )
    return map1, map2


def remap_image(image_bgr: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """Apply an undistort/rectify map to a BGR image."""
    return cv2.remap(
        image_bgr,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def project_points(
    points_cam: np.ndarray,
    K: np.ndarray,
    min_z: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pinhole projection. Returns ``(uv, valid_mask)``."""
    if points_cam.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    z = points_cam[:, 2]
    valid = z > min_z
    uv = np.zeros((points_cam.shape[0], 2), dtype=np.float32)
    if np.any(valid):
        x = points_cam[valid, 0] / z[valid]
        y = points_cam[valid, 1] / z[valid]
        uv[valid, 0] = K[0, 0] * x + K[0, 2]
        uv[valid, 1] = K[1, 1] * y + K[1, 2]
    return uv, valid


def depth_to_colormap(
    depth: np.ndarray,
    depth_min: float,
    depth_max: float,
    invalid_value: float = 0.0,
) -> np.ndarray:
    """Convert depth to Jet RGB (near = red, far = blue)."""
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0) & (d != invalid_value)
    out = np.zeros((*d.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    norm = np.clip((d - depth_min) / (depth_max - depth_min + 1e-8), 0.0, 1.0)
    color_bgr = cv2.applyColorMap(((1.0 - norm) * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    out[valid] = color_rgb[valid]
    return out


def depth_to_pointcloud(
    depth: np.ndarray,
    K: np.ndarray,
    rgb_image: Optional[np.ndarray] = None,
    downsample_factor: int = 4,
    max_points: int = 20000,
    near_plane: float = 0.45,
    far_plane: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Back-project a Z-depth map to an RGB point cloud.

    ``depth`` is rectified-left Z-depth. ``rgb_image`` should be the same camera.

    Returns:
        ``(points_cam (N, 3) float32, colors_rgb (N, 3) uint8)``.
    """
    empty = (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))
    if depth is None or depth.size == 0:
        return empty

    h, w = depth.shape[:2]
    step = max(1, int(downsample_factor))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    depth_ds = depth[::step, ::step]
    u = np.arange(depth_ds.shape[1]) * step + step // 2
    v = np.arange(depth_ds.shape[0]) * step + step // 2
    uu, vv = np.meshgrid(u, v)

    z = depth_ds.reshape(-1).astype(np.float32)
    valid = np.isfinite(z) & (z >= near_plane) & (z <= far_plane)
    if not bool(np.any(valid)):
        return empty

    u_flat = np.clip(uu.reshape(-1)[valid].astype(np.int64), 0, w - 1)
    v_flat = np.clip(vv.reshape(-1)[valid].astype(np.int64), 0, h - 1)
    z_valid = z[valid]
    x = (u_flat - cx) * z_valid / fx
    y = (v_flat - cy) * z_valid / fy
    points = np.stack([x, y, z_valid], axis=-1).astype(np.float32)

    if rgb_image is not None:
        rgb = rgb_image
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        colors = rgb[v_flat, u_flat].astype(np.uint8)
    else:
        lo, hi = float(z_valid.min()), float(z_valid.max())
        norm = (z_valid - lo) / (hi - lo + 1e-8)
        bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET)
        colors = cv2.cvtColor(bgr.reshape(1, -1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)

    if points.shape[0] > max_points:
        idx = np.random.default_rng(0).choice(points.shape[0], max_points, replace=False)
        points, colors = points[idx], colors[idx]
    return points, colors


def draw_hand_skeleton_2d(
    image_rgb: np.ndarray,
    joints_uv: np.ndarray,
    valid: np.ndarray,
    parents: np.ndarray,
    color: Tuple[int, int, int],
    radius: int = 3,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a 2D hand skeleton on an RGB image."""
    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]
    for i, p in enumerate(parents.tolist()):
        if p < 0 or not (valid[i] and valid[p]):
            continue
        p0 = tuple(np.round(joints_uv[p]).astype(int))
        p1 = tuple(np.round(joints_uv[i]).astype(int))
        if 0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h:
            cv2.line(canvas, p0, p1, color, thickness, lineType=cv2.LINE_AA)
    for i, ok in enumerate(valid.tolist()):
        if not ok:
            continue
        pt = tuple(np.round(joints_uv[i]).astype(int))
        if 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(canvas, pt, radius, color, -1, lineType=cv2.LINE_AA)
    return canvas
