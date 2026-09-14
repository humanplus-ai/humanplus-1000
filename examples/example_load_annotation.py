"""Example: list and load a HumanPlus-1000 ``annotation.hdf5``.

Run from the toolkit root:

    python examples/example_load_annotation.py --data_root /path/to/session
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import setup

setup()

from humanplus_viewer.data_loader import (  # noqa: E402
    list_annotation_contents,
    load_release_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List and load a HumanPlus-1000 annotation.hdf5."
    )
    parser.add_argument(
        "--data_root",
        "--session-dir",
        dest="data_root",
        type=str,
        required=True,
        help="Session folder (contains annotation.hdf5)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    annotation_path = data_root / "annotation.hdf5"
    if not annotation_path.is_file():
        print(f"Annotation not found: {annotation_path}")
        return 1

    print("--- annotation.hdf5 contents (top-level) ---")
    contents = list_annotation_contents(annotation_path)
    top = sorted(name for name in contents if "/" not in name)
    for name in top:
        print(f"  {name}: {contents[name]}")

    episode = load_release_episode(
        data_root,
        load_depth=False,
        load_point_cloud=True,
        require_videos=False,
    )
    print("\n--- Loaded data summary ---")
    print(f"  Frames (video timestamps): {episode.num_video_frames}")
    print(f"  Motion frames (sync): {episode.num_motion_frames}")
    print(f"  body_keypoints: {episode.body_keypoints.shape}")
    print(f"  smplh_pose: {episode.body_smplh_pose.shape}")
    print(f"  slam T_slamworld_camera: {episode.slam_T_camera.shape}")
    print(f"  slam point_cloud: {episode.slam_point_cloud.shape}")
    if episode.hand_left_joints_3d is not None:
        print(f"  Hand left joints: {episode.hand_left_joints_3d.shape}")
    if episode.hand_right_joints_3d is not None:
        print(f"  Hand right joints: {episode.hand_right_joints_3d.shape}")

    print("\n--- Calibration ---")
    print(f"  fisheye left K: {episode.calib.fisheye_left_K.shape}")
    print(f"  rectified K: {episode.calib.rectified_K.shape}")
    print(f"  T_head_camera: {episode.calib.T_head_camera.shape}")
    print(f"  T_mocapworld_slamworld: {episode.calib.T_mocapworld_slamworld.shape}")
    print(f"  hamer_camera_side: {episode.calib.hamer_camera_side}")

    print("\n--- Behavior ---")
    print(f"  {episode.task.activity_summarization}")
    print(f"  source: {episode.task.source}")

    print("\nDone. Use these arrays in your own scripts or pass the session to the viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
