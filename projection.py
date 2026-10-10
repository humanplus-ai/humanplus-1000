"""Calibrated body projection onto the rectified right video.

Mesh layers are rasterized with a depth buffer into RGBA at display resolution.
All layers stay separate in Rerun so visibility can be changed during replay.
"""
from __future__ import annotations

import numpy as np
import rerun as rr

from .constants import COLOR_BODY, COLOR_BODY_MESH, PATH_BODY_PROJECTION, SMPL24_PARENTS
from .geometry import invert_T, transform_points


def _rotation(rotation: np.ndarray) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = rotation
    return result


def right_camera_from_world(calib, T_world_slam_camera: np.ndarray, camera_frame: str) -> np.ndarray:
    """Transform mocapworld into rectified-right camera (X right, Y down, Z forward)."""
    right_rect = _rotation(calib.R_rect_right)
    if camera_frame == 'rectified_left':
        right_from_source = right_rect @ calib.T_right_left @ invert_T(_rotation(calib.R_rect_left))
    elif camera_frame == 'rectified_right':
        right_from_source = np.eye(4)
    elif camera_frame in ('left', 'fisheye_left'):
        right_from_source = right_rect @ calib.T_right_left
    elif camera_frame in ('right', 'fisheye_right'):
        right_from_source = right_rect
    else:
        raise ValueError(f'Unsupported SLAM camera_frame for body projection: {camera_frame!r}')
    return right_from_source @ invert_T(T_world_slam_camera)


def scaled_intrinsics(K: np.ndarray, source_wh, target_wh) -> np.ndarray:
    """Use actual resized dimensions, including integer rounding."""
    sw, sh = source_wh
    tw, th = target_wh
    if min(sw, sh, tw, th) <= 0:
        raise ValueError('Projection image dimensions must be positive')
    return np.diag([tw / sw, th / sh, 1.0]) @ np.asarray(K, dtype=np.float64)


def frustum_planes(K: np.ndarray, width: int, height: int, near: float = .05):
    # Each plane is represented by n.p + d >= 0.
    return (
        (np.array([0., 0., 1.]), -near),
        (K[0], 0.), ((width - 1) * K[2] - K[0], 0.),
        (K[1], 0.), ((height - 1) * K[2] - K[1], 0.),
    )


def clip_segment(a: np.ndarray, b: np.ndarray, planes):
    if not np.isfinite([a, b]).all():
        return None
    enter, leave = 0., 1.
    for normal, offset in planes:
        fa, fb = float(normal @ a + offset), float(normal @ b + offset)
        if fa < 0 and fb < 0:
            return None
        if (fa < 0) != (fb < 0):
            crossing = fa / (fa - fb)
            if fa < 0:
                enter = max(enter, crossing)
            else:
                leave = min(leave, crossing)
        if enter > leave:
            return None
    return np.array([a + enter * (b - a), a + leave * (b - a)])


def _pixels(points, K):
    homogeneous = np.asarray(points) @ K.T
    return homogeneous[..., :2] / homogeneous[..., 2:3]


def project_skeleton(joints_camera: np.ndarray, K: np.ndarray, size_wh, near=.05):
    width, height = size_wh
    planes = frustum_planes(K, width, height, near)
    visible = np.isfinite(joints_camera).all(axis=1)
    for normal, offset in planes:
        visible &= joints_camera @ normal + offset >= 0
    points = _pixels(joints_camera[visible], K).reshape(-1, 2)
    lines = []
    for child, parent in enumerate(SMPL24_PARENTS):
        if parent < 0 or child >= len(joints_camera) or parent >= len(joints_camera):
            continue
        clipped = clip_segment(joints_camera[parent], joints_camera[child], planes)
        if clipped is not None:
            lines.append(_pixels(clipped, K))
    return points.astype(np.float32), np.asarray(lines, dtype=np.float32).reshape(-1, 2, 2)


def _clip_polygon(polygon, normal, offset):
    result = []
    previous = polygon[-1]
    fp = float(previous @ normal + offset)
    for current in polygon:
        fc = float(current @ normal + offset)
        if (fp >= 0) != (fc >= 0):
            result.append(previous + (fp / (fp - fc)) * (current - previous))
        if fc >= 0:
            result.append(current)
        previous, fp = current, fc
    return result


def _raster_triangle(triangle, K, rgba, depth, color):
    uv = _pixels(triangle, K)
    height, width = depth.shape
    x0, y0 = np.maximum(np.ceil(uv.min(axis=0) - .5).astype(int), 0)
    x1, y1 = np.minimum(np.floor(uv.max(axis=0) - .5).astype(int), [width - 1, height - 1])
    if x0 > x1 or y0 > y1:
        return
    a, b, c = uv
    denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denominator) < 1e-9:
        return
    y, x = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
    x += .5
    y += .5
    w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
    w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
    w2 = 1 - w0 - w1
    inv_z = w0 / triangle[0, 2] + w1 / triangle[1, 2] + w2 / triangle[2, 2]
    inside = (w0 >= -1e-8) & (w1 >= -1e-8) & (w2 >= -1e-8) & (inv_z > 0)
    z = np.full_like(inv_z, np.inf)
    np.divide(1., inv_z, out=z, where=inside)
    target_depth = depth[y0:y1 + 1, x0:x1 + 1]
    update = inside & (z < target_depth)
    target_depth[update] = z[update]
    target = rgba[y0:y1 + 1, x0:x1 + 1]
    target[update, :3] = color
    target[update, 3] = 255


def render_mesh_rgba(vertices_camera: np.ndarray, faces: np.ndarray, K: np.ndarray, size_wh,
                     near=.05, base_color=COLOR_BODY_MESH):
    """Clip triangles, then use perspective-correct Z for self-occlusion.

    Background is transparent. This intentionally does not infer occlusion by
    scene objects from RGB. No external rendering engine or GPU is required.
    """
    width, height = size_wh
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    planes = frustum_planes(K, width, height, near)
    triangles = np.asarray(vertices_camera, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    valid = np.isfinite(triangles).all(axis=(1, 2))
    fully_inside = valid.copy()
    for normal, offset in planes:
        distances = triangles @ normal + offset
        valid &= np.any(distances >= 0, axis=1)
        fully_inside &= np.all(distances >= 0, axis=1)
    light = np.array([-.3, -.5, -1.])
    light /= np.linalg.norm(light)
    for face_i in np.flatnonzero(valid):
        triangle = triangles[face_i]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        shade = .45 + .55 * abs(float(normal @ light / norm))
        color = np.clip(np.asarray(base_color) * shade, 0, 255).astype(np.uint8)
        polygon = list(triangle)
        if not fully_inside[face_i]:
            for n, d in planes:
                polygon = _clip_polygon(polygon, n, d)
                if len(polygon) < 3:
                    break
        for i in range(1, len(polygon) - 1):
            _raster_triangle(np.array([polygon[0], polygon[i], polygon[i + 1]]), K, rgba, depth, color)
    return rgba


class BodyProjection:
    def __init__(self, calib, camera_frame: str, opacity=.4):
        if not 0 <= opacity <= 1:
            raise ValueError('Projection opacity must be between 0 and 1')
        self.calib, self.camera_frame, self.opacity = calib, camera_frame, opacity
        # Validate the coordinate convention before loading the first frame.
        right_camera_from_world(calib, np.eye(4), camera_frame)

    def log_frame(self, image_rgb, body_world, T_world_slam_camera, vertices_world=None, faces=None):
        height, width = image_rgb.shape[:2]
        K = scaled_intrinsics(self.calib.rectified_K, self.calib.rectified_size, (width, height))
        T = right_camera_from_world(self.calib, T_world_slam_camera, self.camera_frame)
        joints = transform_points(T, body_world)
        points, lines = project_skeleton(joints, K, (width, height))
        rr.log(f'{PATH_BODY_PROJECTION}/image', rr.Image(image_rgb, draw_order=0))
        # Explicit empty batches clear last-frame geometry when it leaves view.
        rr.log(f'{PATH_BODY_PROJECTION}/skeleton/joints', rr.Points2D(
            points, colors=COLOR_BODY, radii=rr.Radius.ui_points(2.5), draw_order=20))
        rr.log(f'{PATH_BODY_PROJECTION}/skeleton/bones', rr.LineStrips2D(
            lines, colors=COLOR_BODY, radii=rr.Radius.ui_points(1.3), draw_order=20))
        if vertices_world is not None and faces is not None:
            rgba = render_mesh_rgba(transform_points(T, vertices_world), faces, K, (width, height))
            rr.log(f'{PATH_BODY_PROJECTION}/mesh', rr.Image(rgba, opacity=self.opacity, draw_order=10))
