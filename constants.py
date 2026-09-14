"""Skeleton topology, colors, and Rerun entity paths."""

from __future__ import annotations

import numpy as np

SMPL24_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)

MANO21_PARENTS = np.array(
    [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
    dtype=np.int32,
)

COLOR_BODY = (90, 160, 255)
COLOR_BODY_MESH = (215, 200, 180)
COLOR_LEFT_HAND = (40, 200, 90)
COLOR_RIGHT_HAND = (180, 70, 220)
COLOR_SLAM_TRAJ = (255, 96, 16)
COLOR_CURRENT_POSE = (255, 32, 0)
COLOR_POINT_CLOUD = (200, 200, 200)

PATH_FISHEYE_LEFT = "cameras/fisheye_left"
PATH_FISHEYE_RIGHT = "cameras/fisheye_right"
PATH_RECTIFIED_LEFT = "cameras/rectified_left"
PATH_RECTIFIED_RIGHT = "cameras/rectified_right"
PATH_DEPTH = "sensors/depth"
PATH_WORLD = "world"
PATH_BODY_SKELETON = f"{PATH_WORLD}/body_skeleton"
PATH_BODY_MESH = f"{PATH_WORLD}/body_mesh"
PATH_SLAM_TRAJ = "slam_traj"
PATH_TASK = "behavior_annotation/description"
