"""Rerun blueprint for the HumanPlus-1000 viewer layout."""

from __future__ import annotations

from typing import Optional, Tuple

import rerun as rr
import rerun.blueprint as rrb

from .constants import (
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


def _bounds2d(width: int, height: int) -> rrb.VisualBounds2D:
    """Lock a 2D view to pixel bounds (no extra auto-fit padding)."""
    return rrb.VisualBounds2D(x_range=[0.0, float(width)], y_range=[0.0, float(height)])


def _image_view(
    name: str,
    origin: str,
    size_wh: Optional[Tuple[int, int]] = None,
) -> rrb.Spatial2DView:
    """Image panel with optional pixel-space visual bounds."""
    kwargs: dict = {"name": name, "origin": origin}
    if size_wh is not None and size_wh[0] > 0 and size_wh[1] > 0:
        kwargs["visual_bounds"] = _bounds2d(size_wh[0], size_wh[1])
    return rrb.Spatial2DView(**kwargs)


def _left_column_share_for_aspect(
    image_aspect: float,
    num_rows: int = 3,
    window_aspect: float = 16.0 / 9.0,
    usable_height_ratio: float = 0.72,
) -> Tuple[int, int]:
    """Estimate left/right column shares from image aspect ratio."""
    left_hw = num_rows / (2.0 * max(image_aspect, 1e-6))
    window_hw = usable_height_ratio / max(window_aspect, 1e-6)
    left_frac = window_hw / max(left_hw, 1e-6)
    left_frac = min(max(left_frac, 0.38), 0.50)
    left = max(1, int(round(left_frac * 10)))
    right = max(1, 10 - left)
    return left, right


def _ground_grid(floor_y: float) -> rrb.LineGrid3D:
    """Y-up ground grid on the ZX plane at ``y = floor_y``."""
    return rrb.LineGrid3D(
        visible=True,
        spacing=0.5,
        plane=rr.components.Plane3D.ZX.with_distance(float(floor_y)),
    )


def _top_down_eye(
    center_xyz: Tuple[float, float, float],
    scene_extent: float,
    floor_y: float,
) -> rrb.EyeControls3D:
    """Top-down camera for the SLAM panel (Y-up world).

    Rerun >= 0.27 accepts ``position`` / ``look_target`` / ``eye_up``.
    0.26 only has ``kind`` / ``speed`` and falls back to Orbital.
    """
    cx, _, cz = center_xyz
    height = max(float(scene_extent) * 1.35, 4.0)
    kwargs = {
        "kind": rrb.Eye3DKind.Orbital,
        "position": [float(cx), float(floor_y) + height, float(cz)],
        "look_target": [float(cx), float(floor_y), float(cz)],
        "eye_up": [0.0, 0.0, -1.0],
    }
    try:
        return rrb.EyeControls3D(**kwargs)
    except TypeError:
        return rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital)


def create_humanplus_blueprint(
    fisheye_wh: Tuple[int, int] = (1920, 1200),
    rectified_wh: Tuple[int, int] = (960, 600),
    depth_wh: Tuple[int, int] = (560, 344),
    image_scale: float = 0.5,
    floor_y: float = 0.0,
    slam_view_center: Optional[Tuple[float, float, float]] = None,
    slam_view_extent: float = 8.0,
    show_slam_point_cloud: bool = False,
    body_repr: str = "both",
) -> rrb.Blueprint:
    """Build the viewer layout.

    Left: fisheye, rectified, depth + SLAM. Right: WORLD x HUMAN and
    Behavior Annotation. Mesh and skeleton are both logged; ``body_repr``
    only sets default visibility (toggle via the Blueprint eye icons).
    """
    fw = max(1, int(round(fisheye_wh[0] * image_scale)))
    fh = max(1, int(round(fisheye_wh[1] * image_scale)))
    rw = max(1, int(round(rectified_wh[0] * image_scale)))
    rh = max(1, int(round(rectified_wh[1] * image_scale)))
    dw, dh = int(depth_wh[0]), int(depth_wh[1])

    aspects = []
    if fh > 0:
        aspects.append(fw / fh)
    if rh > 0:
        aspects.append(rw / rh)
    if dh > 0:
        aspects.append(dw / dh)
    image_aspect = sum(aspects) / len(aspects) if aspects else 1.6
    left_share, right_share = _left_column_share_for_aspect(image_aspect, num_rows=3)
    ground = _ground_grid(floor_y)
    slam_center = slam_view_center or (0.0, float(floor_y), 0.0)
    slam_eye = _top_down_eye(slam_center, slam_view_extent, floor_y)

    fisheye_row = rrb.Horizontal(
        _image_view("FISHEYE CAM 0", PATH_FISHEYE_LEFT, (fw, fh)),
        _image_view("FISHEYE CAM 1", PATH_FISHEYE_RIGHT, (fw, fh)),
    )
    rectified_row = rrb.Horizontal(
        _image_view("RECTIFIED VIEW LEFT", PATH_RECTIFIED_LEFT, (rw, rh)),
        _image_view("RECTIFIED VIEW RIGHT", PATH_RECTIFIED_RIGHT, (rw, rh)),
    )
    depth_slam_row = rrb.Horizontal(
        _image_view("DEPTH", PATH_DEPTH, (dw, dh)),
        rrb.Spatial3DView(
            name="SLAM TRAJECTORY",
            origin=PATH_SLAM_TRAJ,
            background=[245, 245, 245],
            line_grid=ground,
            eye_controls=slam_eye,
        ),
    )
    left_column = rrb.Vertical(
        fisheye_row,
        rectified_row,
        depth_slam_row,
        row_shares=[1, 1, 1],
    )

    world_contents = ["$origin/**"]
    world_overrides: dict = {}
    if not show_slam_point_cloud:
        world_overrides[f"{PATH_WORLD}/slam_point_cloud"] = rrb.EntityBehavior(
            visible=False
        )

    if body_repr == "mesh":
        world_overrides[PATH_BODY_SKELETON] = rrb.EntityBehavior(visible=False)
    elif body_repr == "skeleton":
        world_overrides[PATH_BODY_MESH] = rrb.EntityBehavior(visible=False)

    right_column = rrb.Vertical(
        rrb.Spatial3DView(
            name="WORLD × HUMAN",
            origin=PATH_WORLD,
            contents=world_contents,
            overrides=world_overrides or None,
            background=[28, 30, 34],
            line_grid=ground,
        ),
        rrb.TextDocumentView(name="Behavior Annotation", origin=PATH_TASK),
        row_shares=[5, 2],
    )

    return rrb.Blueprint(
        rrb.Horizontal(
            left_column,
            right_column,
            column_shares=[left_share, right_share],
        ),
        rrb.TimePanel(state="expanded"),
        collapse_panels=False,
    )
