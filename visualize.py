"""HumanPlus-1000 Rerun viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from tqdm import tqdm

from .blueprint import create_humanplus_blueprint
from .body_model import SmplhModel, resolve_smplh_model
from .constants import (
    COLOR_BODY,
    COLOR_BODY_MESH,
    COLOR_CURRENT_POSE,
    COLOR_LEFT_HAND,
    COLOR_POINT_CLOUD,
    COLOR_RIGHT_HAND,
    COLOR_SLAM_TRAJ,
    MANO21_PARENTS,
    PATH_BODY_MESH,
    PATH_BODY_SKELETON,
    PATH_DEPTH,
    PATH_FISHEYE_LEFT,
    PATH_FISHEYE_RIGHT,
    PATH_RECTIFIED_LEFT,
    PATH_RECTIFIED_RIGHT,
    PATH_SLAM_TRAJ,
    PATH_TASK,
    PATH_WORLD,
)
from .data_loader import (
    ReleaseEpisode,
    StereoVideoReader,
    build_video_to_depth_index,
    format_task_markdown,
    load_release_episode,
)
from .geometry import (
    depth_to_colormap,
    depth_to_pointcloud,
    draw_hand_skeleton_2d,
    estimate_floor_y_first_frame,
    estimate_floor_y_from_body,
    invert_T,
    make_fisheye_undistort_maps,
    project_points,
    remap_image,
    smpl24_skeleton_lines,
    transform_points,
)


def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to RGB."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _scale_rgb(image_rgb: np.ndarray, scale: float) -> np.ndarray:
    """Resize an RGB image by ``scale``."""
    if abs(scale - 1.0) < 1e-6:
        return image_rgb
    h, w = image_rgb.shape[:2]
    return cv2.resize(
        image_rgb,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _resolve_hand_joints_camera(
    joints_3d: Optional[np.ndarray],
    wrist_position_camera: Optional[np.ndarray],
    valid: bool = True,
) -> Optional[np.ndarray]:
    """HaMeR-camera MANO-21 joints: ``joints_cam = joints_local + wrist``.

    ``valid`` is unused for visibility; any finite local joints + wrist are drawn
    so sparse HaMeR flags do not flicker the overlay.
    """
    del valid
    if joints_3d is None or wrist_position_camera is None:
        return None
    joints_local = np.asarray(joints_3d, dtype=np.float64)
    wrist = np.asarray(wrist_position_camera, dtype=np.float64).reshape(3)
    if not np.isfinite(wrist).all():
        return None
    finite = np.isfinite(joints_local).all(axis=1)
    if not bool(np.any(finite)):
        return None
    out = joints_local.astype(np.float32).copy()
    out[finite] = (joints_local[finite] + wrist).astype(np.float32)
    out[~finite] = np.nan
    return out


def _hold_hand_joints(
    current: Optional[np.ndarray],
    state: dict,
    motion_i: int,
    max_hold_frames: int = 12,
) -> Optional[np.ndarray]:
    """Hold the last HaMeR observation across sparse gaps (typically 1-in-3)."""
    if current is not None:
        state["joints"] = current
        state["motion_i"] = int(motion_i)
        return current
    prev = state.get("joints")
    prev_i = state.get("motion_i")
    if prev is not None and prev_i is not None:
        if int(motion_i) - int(prev_i) <= int(max_hold_frames):
            return prev
    state.clear()
    return None


def _rotation_to_T(rotation: np.ndarray) -> np.ndarray:
    """Pad a ``3x3`` rotation to a ``4x4`` rigid transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return T


def _hamer_to_rectified_transforms(calib) -> Tuple[np.ndarray, np.ndarray]:
    """HaMeR raw-camera frame → left/right rectified camera frames.

    ``wrist_position_camera`` lives in the physical HaMeR-eye fisheye frame.
    Rectified images are produced with ``R_rect_*``; match eye first, then rectify.
    """
    T_right_left = np.asarray(calib.T_right_left, dtype=np.float64)
    T_left_right = invert_T(T_right_left)
    if calib.hamer_camera_side == "left":
        T_left_hamer = np.eye(4, dtype=np.float64)
    elif calib.hamer_camera_side == "right":
        T_left_hamer = T_left_right
    else:
        raise ValueError(f"Unknown hamer_camera_side={calib.hamer_camera_side!r}")
    T_right_hamer = T_right_left @ T_left_hamer
    T_rect_left_hamer = _rotation_to_T(calib.R_rect_left) @ T_left_hamer
    T_rect_right_hamer = _rotation_to_T(calib.R_rect_right) @ T_right_hamer
    return T_rect_left_hamer, T_rect_right_hamer


def _overlay_hands_on_rectified(
    image_rgb: np.ndarray,
    K: np.ndarray,
    left_joints_cam: Optional[np.ndarray],
    right_joints_cam: Optional[np.ndarray],
    T_view_hamercam: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Project HaMeR-camera hand joints onto a rectified view."""
    canvas = image_rgb.copy()
    T = np.eye(4, dtype=np.float64) if T_view_hamercam is None else T_view_hamercam

    def _draw(joints: Optional[np.ndarray], color: Tuple[int, int, int]) -> None:
        nonlocal canvas
        if joints is None or joints.shape[0] == 0:
            return
        finite = np.isfinite(joints).all(axis=1)
        if not bool(np.any(finite)):
            return
        filled = np.where(np.isfinite(joints), joints, 0.0).astype(np.float32)
        pts_view = transform_points(T, filled)
        uv, proj_ok = project_points(pts_view, K)
        valid = finite & proj_ok
        h, w = canvas.shape[:2]
        if joints.shape[0] == len(MANO21_PARENTS):
            canvas = draw_hand_skeleton_2d(canvas, uv, valid, MANO21_PARENTS, color)
            return
        for i, ok in enumerate(valid.tolist()):
            if not ok:
                continue
            pt = tuple(np.round(uv[i]).astype(int))
            if 0 <= pt[0] < w and 0 <= pt[1] < h:
                cv2.circle(canvas, pt, 6 if joints.shape[0] == 1 else 3, color, -1)

    _draw(left_joints_cam, COLOR_LEFT_HAND)
    _draw(right_joints_cam, COLOR_RIGHT_HAND)
    return canvas


def _scene_center_extent(points: np.ndarray) -> Tuple[Tuple[float, float, float], float]:
    """Horizontal center and extent of a point cloud (for the SLAM top-down view)."""
    if points.size == 0:
        return (0.0, 0.0, 0.0), 8.0
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = float(max(hi[0] - lo[0], hi[2] - lo[2], 1.0))
    return (float(center[0]), float(center[1]), float(center[2])), extent


def _resolve_floor_y(
    body_keypoints: np.ndarray,
    *,
    ground_mode: str = "first_frame",
    body_y_offset: float = 0.0,
    first_frame_mesh_y: Optional[float] = None,
) -> float:
    """Y of the WORLD / SLAM ground grid.

    ``first_frame``: first-frame sole (mesh min-Y if available, else feet).
    ``fixed``: mocapworld ``y = 0``.
    ``feet``: 5th percentile of foot keypoints over the clip.
    """
    mode = str(ground_mode).lower()
    offset = float(body_y_offset)
    if mode == "fixed":
        return 0.0
    if mode == "feet":
        return estimate_floor_y_from_body(body_keypoints) + offset
    if mode in ("first_frame", "first-frame"):
        if first_frame_mesh_y is not None and np.isfinite(first_frame_mesh_y):
            return float(first_frame_mesh_y)
        return estimate_floor_y_first_frame(body_keypoints) + offset
    raise ValueError(
        f"Unknown ground_mode={ground_mode!r}; expected first_frame / fixed / feet"
    )


def _log_static(episode: ReleaseEpisode, T_mocap_slam: np.ndarray) -> None:
    """Log Y-up coordinates, SLAM map points, and the task document."""
    rr.log(PATH_WORLD, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.log(PATH_SLAM_TRAJ, rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
    rr.log(
        PATH_TASK,
        rr.TextDocument(
            format_task_markdown(episode.task),
            media_type=rr.MediaType.MARKDOWN,
        ),
        static=True,
    )

    pc_mocap = transform_points(T_mocap_slam, episode.slam_point_cloud)
    if pc_mocap.shape[0] == 0:
        return
    rr.log(
        f"{PATH_WORLD}/slam_point_cloud",
        rr.Points3D(pc_mocap, colors=COLOR_POINT_CLOUD, radii=0.004),
        static=True,
    )
    step = max(1, pc_mocap.shape[0] // 20000)
    rr.log(
        f"{PATH_SLAM_TRAJ}/map_points",
        rr.Points3D(pc_mocap[::step], colors=(170, 170, 170), radii=0.003),
        static=True,
    )


def _log_small_camera_frustum(
    entity_path: str,
    T_mocap_cam: np.ndarray,
    size_m: float = 0.10,
    line_radius: float = 0.005,
) -> None:
    """Log a fixed-size wireframe frustum (RDF: +X right, +Y down, +Z forward)."""
    half_w = 0.55 * size_m
    half_h = 0.40 * size_m
    depth = float(size_m)
    corners_cam = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [-half_w, -half_h, depth],
            [half_w, -half_h, depth],
            [half_w, half_h, depth],
            [-half_w, half_h, depth],
        ],
        dtype=np.float64,
    )
    R = T_mocap_cam[:3, :3].astype(np.float64)
    t = T_mocap_cam[:3, 3].astype(np.float64)
    corners = (corners_cam @ R.T) + t
    o, a, b, c, d = [p.astype(np.float32) for p in corners]
    strips = [
        np.stack([o, a]),
        np.stack([o, b]),
        np.stack([o, c]),
        np.stack([o, d]),
        np.stack([a, b]),
        np.stack([b, c]),
        np.stack([c, d]),
        np.stack([d, a]),
    ]
    rr.log(
        entity_path,
        rr.LineStrips3D(strips, colors=[COLOR_CURRENT_POSE] * len(strips), radii=line_radius),
    )
    rr.log(
        f"{entity_path}/anchor",
        rr.Points3D([o], colors=COLOR_CURRENT_POSE, radii=max(0.02, 0.18 * size_m)),
    )


def _setup_rerun_sinks(
    app_id: str,
    spawn: bool,
    output_rrd: Optional[str | Path],
) -> Optional[Path]:
    """Connect Rerun to a live viewer and/or an optional ``.rrd`` file.

    ``.rrd`` is Rerun's on-disk recording. ``rr.save()`` replaces other sinks, so
    saving while spawning must use ``set_sinks(GrpcSink, FileSink)`` or the
    viewer stays empty.
    """
    rr.init(app_id, spawn=False)
    out_path: Optional[Path] = None
    if output_rrd is not None:
        out_path = Path(output_rrd)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if spawn:
        rr.spawn()
        print("[viewer] opened Rerun window")
    if spawn and out_path is not None:
        rr.set_sinks(rr.GrpcSink(), rr.FileSink(str(out_path)))
        print(f"[viewer] also writing recording {out_path}")
    elif out_path is not None:
        rr.save(str(out_path))
        print(f"[viewer] writing {out_path} (no window). Open later with: rerun {out_path}")
    elif not spawn:
        raise ValueError("Pass --output-rrd when using --no-spawn, or omit --no-spawn to open the viewer.")
    return out_path


def visualize_release(
    release_dir: str | Path,
    output_rrd: Optional[str | Path] = None,
    spawn: bool = True,
    max_frames: int = -1,
    stride: int = 1,
    image_scale: float = 0.5,
    task_json: Optional[str | Path] = None,
    point_cloud_max_points: int = 80000,
    depth_rgb_points: int = 20000,
    depth_rgb_downsample: int = 4,
    show_slam_point_cloud: bool = False,
    body_repr: str = "both",
    smplh_model_path: Optional[str | Path] = None,
    smplh_gender: str = "male",
    show_world_camera_frustum: bool = False,
    body_y_offset: float = 0.0,
    ground_mode: str = "first_frame",
    app_id: str = "HumanPlus-1000",
) -> Optional[Path]:
    """Visualize one HumanPlus-1000 session in Rerun.

    Both mesh and skeleton are logged. ``body_repr`` only sets default
    visibility; toggle entities with the Blueprint eye icons.
    """
    episode = load_release_episode(
        release_dir,
        task_json=task_json,
        load_depth=True,
        load_point_cloud=True,
        point_cloud_max_points=point_cloud_max_points,
    )
    calib = episode.calib
    print(
        f"[viewer] videos: {episode.video_left_path.name} / "
        f"{episode.video_right_path.name} | hamer={calib.hamer_camera_side} "
        f"left_half={calib.left_half} rect={calib.rectification_source}"
    )
    T_mocap_slam = calib.T_mocapworld_slamworld.astype(np.float64)
    body_shift = np.asarray([0.0, float(body_y_offset), 0.0], dtype=np.float32)
    pc_preview = transform_points(T_mocap_slam, episode.slam_point_cloud)
    slam_center, slam_extent = _scene_center_extent(pc_preview)

    depth_h, depth_w = (0, 0)
    if episode.depth is not None and episode.depth.ndim >= 2:
        depth_h = int(episode.depth.shape[-2])
        depth_w = int(episode.depth.shape[-1])

    smplh_model: Optional[SmplhModel] = None
    if body_repr in ("mesh", "both"):
        resolved = resolve_smplh_model(smplh_model_path, smplh_gender)
        if resolved is None:
            print(
                "[viewer] SMPL-H model.npz not found; falling back to skeleton. "
                "Pass --smplh-model or set HUMANPLUS_SMPLH_MODEL."
            )
            body_repr = "skeleton"
        else:
            smplh_model = SmplhModel(resolved)
            print(f"[viewer] SMPL-H mesh: {resolved}")

    first_frame_mesh_y: Optional[float] = None
    if (
        str(ground_mode).lower() in ("first_frame", "first-frame")
        and smplh_model is not None
        and episode.body_keypoints.shape[0] > 0
        and episode.body_smplh_pose.shape[0] > 0
    ):
        body0 = episode.body_keypoints[0] + body_shift
        verts0, _ = smplh_model.forward_at_pelvis(episode.body_smplh_pose[0], body0[0])
        ys = verts0[:, 1]
        ys = ys[np.isfinite(ys)]
        if ys.size > 0:
            first_frame_mesh_y = float(np.min(ys))

    floor_y = _resolve_floor_y(
        episode.body_keypoints,
        ground_mode=ground_mode,
        body_y_offset=body_y_offset,
        first_frame_mesh_y=first_frame_mesh_y,
    )
    mode = str(ground_mode).lower()
    if mode == "fixed":
        print("[viewer] ground grid: mocapworld y=0 (fixed)")
    elif mode == "feet":
        print(f"[viewer] ground grid: foot 5th percentile floor_y={floor_y:+.4f} m (feet)")
    elif first_frame_mesh_y is not None:
        print(
            f"[viewer] ground grid: first-frame mesh min-y floor_y={floor_y:+.4f} m "
            "(first_frame)"
        )
    else:
        print(
            f"[viewer] ground grid: first-frame foot joints floor_y={floor_y:+.4f} m "
            "(first_frame)"
        )

    out_path = _setup_rerun_sinks(app_id, spawn=spawn, output_rrd=output_rrd)
    rr.send_blueprint(
        create_humanplus_blueprint(
            fisheye_wh=calib.fisheye_left_size,
            rectified_wh=calib.rectified_size,
            depth_wh=(depth_w, depth_h) if depth_w > 0 else (560, 344),
            image_scale=image_scale,
            floor_y=floor_y,
            slam_view_center=slam_center,
            slam_view_extent=slam_extent,
            show_slam_point_cloud=show_slam_point_cloud,
            body_repr=body_repr,
        )
    )

    map_l1, map_l2 = make_fisheye_undistort_maps(
        calib.fisheye_left_K,
        calib.fisheye_left_D,
        calib.fisheye_left_size,
        calib.rectified_K,
        calib.rectified_size,
        R=calib.R_rect_left,
    )
    map_r1, map_r2 = make_fisheye_undistort_maps(
        calib.fisheye_right_K,
        calib.fisheye_right_D,
        calib.fisheye_right_size,
        calib.rectified_K,
        calib.rectified_size,
        R=calib.R_rect_right,
    )
    T_rect_left_hamer, T_rect_right_hamer = _hamer_to_rectified_transforms(calib)

    depth_lookup = build_video_to_depth_index(
        episode.depth_frame_indices, episode.num_video_frames
    )
    depth_min = float(episode.depth_attrs.get("min_depth_m", 0.1))
    depth_max = float(episode.depth_attrs.get("max_depth_m", 8.0))
    depth_vis_max = min(depth_max, 4.0)

    _log_static(episode, T_mocap_slam)
    if smplh_model is not None:
        rr.log(
            PATH_BODY_MESH,
            rr.Mesh3D.from_fields(
                triangle_indices=smplh_model.faces,
                albedo_factor=COLOR_BODY_MESH,
            ),
            static=True,
        )
    traj_mocap = transform_points(T_mocap_slam, episode.slam_translation)

    n = episode.num_motion_frames
    if max_frames > 0:
        n = min(n, max_frames)
    indices = list(range(0, n, max(1, stride)))
    last_depth_rgb = None
    hand_hold_left: dict = {}
    hand_hold_right: dict = {}
    hand_hold_frames = max(12, 4 * max(1, stride))

    with StereoVideoReader(episode.video_left_path, episode.video_right_path) as videos:
        for motion_i in tqdm(indices, desc="HumanPlus Viewer"):
            video_i = int(episode.sync_video_frame_indices[motion_i])
            if video_i < 0 or video_i >= episode.num_video_frames:
                continue

            rr.set_time("frame", sequence=video_i)
            rr.set_time("stable_time", duration=float(video_i) / 30.0)

            left_bgr, right_bgr = videos.read_index(video_i)
            rr.log(
                PATH_FISHEYE_LEFT,
                rr.Image(_scale_rgb(_bgr_to_rgb(left_bgr), image_scale)),
            )
            rr.log(
                PATH_FISHEYE_RIGHT,
                rr.Image(_scale_rgb(_bgr_to_rgb(right_bgr), image_scale)),
            )

            rect_l = remap_image(left_bgr, map_l1, map_l2)
            rect_r = remap_image(right_bgr, map_r1, map_r2)

            T_slam_cam = episode.slam_T_camera[video_i].astype(np.float64)
            T_mocap_cam = T_mocap_slam @ T_slam_cam

            left_wrist_cam = (
                episode.hand_left_wrist_position_camera[motion_i]
                if episode.hand_left_wrist_position_camera is not None
                else None
            )
            right_wrist_cam = (
                episode.hand_right_wrist_position_camera[motion_i]
                if episode.hand_right_wrist_position_camera is not None
                else None
            )
            left_hand_cam = _hold_hand_joints(
                _resolve_hand_joints_camera(
                    episode.hand_left_joints_3d[motion_i]
                    if episode.hand_left_joints_3d is not None
                    else None,
                    left_wrist_cam,
                    bool(episode.hand_left_valid[motion_i]),
                ),
                hand_hold_left,
                motion_i,
                max_hold_frames=hand_hold_frames,
            )
            right_hand_cam = _hold_hand_joints(
                _resolve_hand_joints_camera(
                    episode.hand_right_joints_3d[motion_i]
                    if episode.hand_right_joints_3d is not None
                    else None,
                    right_wrist_cam,
                    bool(episode.hand_right_valid[motion_i]),
                ),
                hand_hold_right,
                motion_i,
                max_hold_frames=hand_hold_frames,
            )

            rect_l_vis = _overlay_hands_on_rectified(
                _bgr_to_rgb(rect_l),
                calib.rectified_K,
                left_hand_cam,
                right_hand_cam,
                T_view_hamercam=T_rect_left_hamer,
            )
            rect_r_vis = _overlay_hands_on_rectified(
                _bgr_to_rgb(rect_r),
                calib.rectified_K,
                left_hand_cam,
                right_hand_cam,
                T_view_hamercam=T_rect_right_hamer,
            )
            rr.log(PATH_RECTIFIED_LEFT, rr.Image(_scale_rgb(rect_l_vis, image_scale)))
            rr.log(PATH_RECTIFIED_RIGHT, rr.Image(_scale_rgb(rect_r_vis, image_scale)))

            depth_i = int(depth_lookup[video_i]) if video_i < len(depth_lookup) else -1
            depth_frame = (
                episode.depth[depth_i]
                if depth_i >= 0 and episode.depth is not None
                else None
            )
            if depth_frame is not None:
                last_depth_rgb = depth_to_colormap(depth_frame, depth_min, depth_vis_max)
            if last_depth_rgb is not None:
                rr.log(PATH_DEPTH, rr.Image(last_depth_rgb))

            if depth_frame is not None and depth_rgb_points > 0:
                pts_cam, pts_rgb = depth_to_pointcloud(
                    depth_frame,
                    episode.depth_intrinsics
                    if episode.depth_intrinsics is not None
                    else calib.rectified_K,
                    rgb_image=_bgr_to_rgb(rect_l),
                    downsample_factor=depth_rgb_downsample,
                    max_points=depth_rgb_points,
                    near_plane=depth_min,
                    far_plane=depth_vis_max,
                )
                if pts_cam.shape[0] > 0:
                    rr.log(
                        f"{PATH_WORLD}/depth_points",
                        rr.Points3D(
                            transform_points(T_mocap_cam, pts_cam),
                            colors=pts_rgb,
                            radii=0.006,
                        ),
                    )

            body = episode.body_keypoints[motion_i] + body_shift
            rr.log(
                f"{PATH_BODY_SKELETON}/joints",
                rr.Points3D(body, colors=COLOR_BODY, radii=0.02),
            )
            body_lines = smpl24_skeleton_lines(body)
            if body_lines.shape[0] > 0:
                rr.log(
                    f"{PATH_BODY_SKELETON}/lines",
                    rr.LineStrips3D(body_lines, colors=[COLOR_BODY], radii=0.01),
                )

            if smplh_model is not None:
                verts, normals = smplh_model.forward_at_pelvis(
                    episode.body_smplh_pose[motion_i], body[0]
                )
                rr.log(
                    PATH_BODY_MESH,
                    rr.Mesh3D.from_fields(
                        vertex_positions=verts, vertex_normals=normals
                    ),
                )

            if show_world_camera_frustum:
                _log_small_camera_frustum(
                    f"{PATH_WORLD}/camera_frustum",
                    T_mocap_cam,
                    size_m=0.12,
                    line_radius=0.005,
                )
            _log_small_camera_frustum(
                f"{PATH_SLAM_TRAJ}/camera_frustum",
                T_mocap_cam,
                size_m=0.45,
                line_radius=0.012,
            )

            if traj_mocap.shape[0] > 0:
                end = min(video_i, traj_mocap.shape[0] - 1)
                trail = traj_mocap[: end + 1]
                if trail.shape[0] > 2500:
                    idx = np.linspace(0, trail.shape[0] - 1, 2500).astype(np.int64)
                    trail = trail[idx]
                rr.log(
                    f"{PATH_SLAM_TRAJ}/pose_trail",
                    rr.Points3D(trail, colors=COLOR_SLAM_TRAJ, radii=0.015),
                )

    return out_path


visualize_session = visualize_release


def build_argparser() -> argparse.ArgumentParser:
    """CLI for the HumanPlus-1000 viewer."""
    parser = argparse.ArgumentParser(
        description="Visualize a HumanPlus-1000 session with Rerun.",
    )
    parser.add_argument(
        "--session-dir",
        "--release-dir",
        "--data-root",
        dest="session_dir",
        type=str,
        required=True,
        help="Session folder (annotation.hdf5 + fisheye mp4)",
    )
    parser.add_argument(
        "--output-rrd",
        type=str,
        default=None,
        help="Optional Rerun recording (.rrd) on disk; the live window still opens unless --no-spawn",
    )
    parser.add_argument(
        "--no-spawn",
        action="store_true",
        help="Do not open the Rerun window (use with --output-rrd)",
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--task-json", type=str, default=None)
    parser.add_argument("--point-cloud-max-points", type=int, default=80000)
    parser.add_argument(
        "--depth-rgb-points",
        type=int,
        default=20000,
        help="Max RGB depth points per frame; 0 disables",
    )
    parser.add_argument("--depth-rgb-downsample", type=int, default=4)
    parser.add_argument(
        "--show-slam-point-cloud",
        action="store_true",
        help="Show the grey SLAM map in WORLD (hidden by default; toggle in Blueprint)",
    )
    parser.add_argument(
        "--body-repr",
        choices=("mesh", "skeleton", "both"),
        default="both",
        help="Default WORLD body: mesh / skeleton / both (toggle with Blueprint eyes)",
    )
    parser.add_argument(
        "--smplh-model",
        type=str,
        default=None,
        help="SMPL-H model.npz; otherwise auto-search / HUMANPLUS_SMPLH_MODEL",
    )
    parser.add_argument(
        "--smplh-gender", choices=("male", "female", "neutral"), default="male"
    )
    parser.add_argument(
        "--body-y-offset",
        type=float,
        default=0.0,
        help="Vertical body offset in meters (display only)",
    )
    parser.add_argument(
        "--ground-mode",
        choices=("first_frame", "fixed", "feet"),
        default="first_frame",
        help="Ground grid: first_frame (default), fixed (y=0), or feet (clip percentile)",
    )
    parser.add_argument(
        "--show-world-frustum",
        action="store_true",
        help="Draw a camera frustum in the WORLD view (off by default)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    args = build_argparser().parse_args(argv)
    visualize_release(
        release_dir=args.session_dir,
        output_rrd=args.output_rrd,
        spawn=not args.no_spawn,
        max_frames=args.max_frames,
        stride=args.stride,
        image_scale=args.image_scale,
        task_json=args.task_json,
        point_cloud_max_points=args.point_cloud_max_points,
        depth_rgb_points=args.depth_rgb_points,
        depth_rgb_downsample=args.depth_rgb_downsample,
        show_slam_point_cloud=args.show_slam_point_cloud,
        body_repr=args.body_repr,
        smplh_model_path=args.smplh_model,
        smplh_gender=args.smplh_gender,
        show_world_camera_frustum=args.show_world_frustum,
        body_y_offset=args.body_y_offset,
        ground_mode=args.ground_mode,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
