"""Load a HumanPlus-1000 session from ``annotation.hdf5`` and stereo video."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import h5py
import numpy as np


def _decode(val: Any) -> Any:
    """Decode HDF5 bytes / scalar values to Python objects."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, np.ndarray) and val.shape == ():
        return _decode(val.item())
    return val


def _optional_dataset(group: h5py.Group, path: str) -> Optional[np.ndarray]:
    """Return a dataset as ``ndarray``, or ``None`` if the path is missing."""
    if path not in group:
        return None
    return np.asarray(group[path][...])


@dataclass
class ReleaseCalibration:
    """Camera calibration and world-frame transforms for one session."""

    fisheye_left_K: np.ndarray
    fisheye_left_D: np.ndarray
    fisheye_left_size: Tuple[int, int]
    fisheye_right_K: np.ndarray
    fisheye_right_D: np.ndarray
    fisheye_right_size: Tuple[int, int]
    rectified_K: np.ndarray
    rectified_size: Tuple[int, int]
    T_right_left: np.ndarray
    T_head_camera: np.ndarray
    T_mocapworld_slamworld: np.ndarray
    R_rect_left: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    R_rect_right: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    hamer_camera_side: str = "left"
    left_half: int = 0
    rectification_source: str = "identity_fallback"


@dataclass
class TaskDescription:
    """Behavior caption loaded from HDF5 or a sidecar JSON file."""

    activity_summarization: str = "N/A"
    source: str = "placeholder"


@dataclass
class ReleaseEpisode:
    """One HumanPlus-1000 session (annotation + video paths + arrays)."""

    root: Path
    attrs: Dict[str, Any]
    calib: ReleaseCalibration
    video_left_path: Path
    video_right_path: Path
    video_timestamp_ns: np.ndarray
    video_frame_indices: np.ndarray
    sync_utc_ns: np.ndarray
    sync_video_frame_indices: np.ndarray
    sync_video_time_error_ns: np.ndarray
    sync_depth_frame_indices: np.ndarray
    body_keypoints: np.ndarray
    body_T_root: np.ndarray
    body_smplh_pose: np.ndarray
    body_foot_contact: np.ndarray
    body_frame_indices: np.ndarray
    body_utc_ns: np.ndarray
    hand_left_T_wrist: np.ndarray
    hand_right_T_wrist: np.ndarray
    hand_left_pose: np.ndarray
    hand_right_pose: np.ndarray
    hand_left_valid: np.ndarray
    hand_right_valid: np.ndarray
    hand_left_confidence: np.ndarray
    hand_right_confidence: np.ndarray
    hand_left_joints_3d: Optional[np.ndarray] = None
    hand_right_joints_3d: Optional[np.ndarray] = None
    hand_left_wrist_position_camera: Optional[np.ndarray] = None
    hand_right_wrist_position_camera: Optional[np.ndarray] = None
    slam_T_camera: np.ndarray = field(default_factory=lambda: np.zeros(0))
    slam_translation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    slam_quaternion: np.ndarray = field(default_factory=lambda: np.zeros(0))
    slam_point_cloud: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    slam_frame_indices: np.ndarray = field(default_factory=lambda: np.zeros(0))
    slam_utc_ns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    depth: Optional[np.ndarray] = None
    depth_frame_indices: np.ndarray = field(default_factory=lambda: np.zeros(0))
    depth_utc_ns: np.ndarray = field(default_factory=lambda: np.zeros(0))
    depth_intrinsics: Optional[np.ndarray] = None
    depth_attrs: Dict[str, Any] = field(default_factory=dict)
    task: TaskDescription = field(default_factory=TaskDescription)
    unavailable_products: Tuple[str, ...] = ()

    @property
    def num_motion_frames(self) -> int:
        """Length of the synchronized motion / annotation timeline."""
        return int(self.sync_utc_ns.shape[0])

    @property
    def num_video_frames(self) -> int:
        """Number of egocentric video frames."""
        if self.video_timestamp_ns.shape[0] > 0:
            return int(self.video_timestamp_ns.shape[0])
        return int(self.video_frame_indices.shape[0])


def _load_wrist_position_camera_from_hamer_jsonl(
    release_root: Path,
    sync_video_frame_indices: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load HaMeR wrist translations from a sidecar ``hamer_hand_tracks.jsonl``.

    HDF5 ``joints_3d`` is MANO-wrist-local. 2D overlay uses
    ``joints_cam = joints_local + wrist_position_camera``.
    """
    candidates = [
        release_root / "hamer_hand_tracks.jsonl",
        release_root.parent / "hamer_hand_tracks.jsonl",
    ]
    jsonl_path = next((p for p in candidates if p.is_file()), None)
    if jsonl_path is None:
        return None, None

    by_formal: Dict[int, Dict[str, np.ndarray]] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            formal = obj.get("formal_frame_index", obj.get("frame_index"))
            if formal is None:
                continue
            formal_i = int(formal)
            sides = (obj.get("hamer") or {}).get("sides") or {}
            entry: Dict[str, np.ndarray] = {}
            for side in ("left", "right"):
                payload = sides.get(side) or {}
                wrist = payload.get("wrist_position_camera")
                if payload.get("valid") and wrist is not None:
                    entry[side] = np.asarray(wrist, dtype=np.float32).reshape(3)
            if entry:
                by_formal[formal_i] = entry

    n = int(sync_video_frame_indices.shape[0])
    left = np.full((n, 3), np.nan, dtype=np.float32)
    right = np.full((n, 3), np.nan, dtype=np.float32)
    hit = 0
    for i, vi in enumerate(np.asarray(sync_video_frame_indices).tolist()):
        item = by_formal.get(int(vi))
        if item is None:
            continue
        hit += 1
        if "left" in item:
            left[i] = item["left"]
        if "right" in item:
            right[i] = item["right"]
    if hit == 0:
        return None, None
    return left, right


def _find_sidecar_json(release_dir: Path, names: Tuple[str, ...]) -> Optional[Path]:
    """Find a sidecar JSON in the session directory or its parent."""
    for base in (release_dir, release_dir.parent):
        for name in names:
            path = base / name
            if path.is_file():
                return path
    return None


def _resolve_hamer_camera_side(
    release_dir: Path,
    left_half: int,
    hdf5_side: Optional[str] = None,
) -> str:
    """Resolve which physical camera HaMeR observed (``left`` or ``right``).

    ``eye`` / ``eye_index`` in sidecar calibration refers to a side-by-side
    half-image (0 = left half), which is not always the physical left eye.
    Physical eye is determined by ``stereo_calibration.eye_layout.left_half``.
    """
    if hdf5_side in ("left", "right"):
        return str(hdf5_side)

    smpl_path = _find_sidecar_json(
        release_dir,
        ("camera_smpl_calibration.json", "camera_calibration.json"),
    )
    eye_index = 0
    if smpl_path is not None:
        document = json.loads(smpl_path.read_text(encoding="utf-8"))
        if document.get("eye_index") is not None:
            eye_index = int(document["eye_index"])
        else:
            eye_name = str(document.get("eye", "left")).lower()
            eye_index = 0 if eye_name.startswith("l") else 1
    return "left" if eye_index == int(left_half) else "right"


def _load_rectification_extras(
    f: h5py.File,
    release_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, str, int, str]:
    """Load stereo rectification rotations and the HaMeR camera side."""
    g = f["calibration"]
    rect = g["rectified_stereo"]
    R_left = np.eye(3, dtype=np.float64)
    R_right = np.eye(3, dtype=np.float64)
    left_half = 0
    source = "identity_fallback"
    hdf5_side = None

    if "R_rect_left" in rect and "R_rect_right" in rect:
        R_left = np.asarray(rect["R_rect_left"][...], dtype=np.float64).reshape(3, 3)
        R_right = np.asarray(rect["R_rect_right"][...], dtype=np.float64).reshape(3, 3)
        source = "annotation.hdf5"
        if "left_half" in rect.attrs:
            left_half = int(rect.attrs["left_half"])
        if "hamer_camera_side" in rect.attrs:
            hdf5_side = _decode(rect.attrs["hamer_camera_side"])
    elif "rotation_left" in rect and "rotation_right" in rect:
        R_left = np.asarray(rect["rotation_left"][...], dtype=np.float64).reshape(3, 3)
        R_right = np.asarray(rect["rotation_right"][...], dtype=np.float64).reshape(3, 3)
        source = "annotation.hdf5"
        if "left_half" in rect.attrs:
            left_half = int(rect.attrs["left_half"])
        if "hamer_camera_side" in rect.attrs:
            hdf5_side = _decode(rect.attrs["hamer_camera_side"])
    else:
        stereo_path = _find_sidecar_json(
            release_dir,
            ("stereo_calibration.json", "stereo_calib.json"),
        )
        if stereo_path is not None:
            document = json.loads(stereo_path.read_text(encoding="utf-8"))
            rectification = document["rectification"]
            R_left = np.asarray(rectification["rotation_left"], dtype=np.float64).reshape(3, 3)
            R_right = np.asarray(rectification["rotation_right"], dtype=np.float64).reshape(3, 3)
            left_half = int(document.get("eye_layout", {}).get("left_half", 0))
            source = str(stereo_path)

    hamer_side = _resolve_hamer_camera_side(release_dir, left_half, hdf5_side)
    return R_left, R_right, hamer_side, left_half, source


def _int_dataset_or_arange(group: h5py.Group, path: str, length: int) -> np.ndarray:
    """Read an int64 index dataset, or ``0..length-1`` if it is absent."""
    if path in group:
        return np.asarray(group[path][...], dtype=np.int64)
    return np.arange(length, dtype=np.int64)


def _video_filename(video_g: h5py.Group, key: str, default: str) -> str:
    """Read ``video/fisheye_*_file`` from a dataset or attribute."""
    if key in video_g:
        name = str(_decode(video_g[key][()])).strip()
        if name:
            return name
    if key in video_g.attrs:
        name = str(_decode(video_g.attrs[key])).strip()
        if name:
            return name
    return default


def _load_mocapworld_slamworld(calibration: h5py.Group) -> np.ndarray:
    """Load slamworld → mocapworld.

    Prefer ``T_mocapworld_slamworld_scene`` when present (older files);
    otherwise use ``T_mocapworld_slamworld``. Point clouds, depth, and
    camera trajectories all share this transform.
    """
    if "T_mocapworld_slamworld_scene" in calibration:
        return np.asarray(
            calibration["T_mocapworld_slamworld_scene"][...], dtype=np.float64
        )
    return np.asarray(
        calibration["T_mocapworld_slamworld"][...], dtype=np.float64
    )


def _load_calibration(f: h5py.File, release_dir: Path) -> ReleaseCalibration:
    """Load ``calibration/`` plus rectification extras."""
    g = f["calibration"]
    left = g["fisheye_left"]
    right = g["fisheye_right"]
    rect = g["rectified_stereo"]
    ls = [int(x) for x in np.asarray(left["image_size"][...])]
    rs = [int(x) for x in np.asarray(right["image_size"][...])]
    rect_s = [int(x) for x in np.asarray(rect["image_size"][...])]
    R_left, R_right, hamer_side, left_half, source = _load_rectification_extras(
        f, release_dir
    )
    return ReleaseCalibration(
        fisheye_left_K=np.asarray(left["intrinsics"][...], dtype=np.float64),
        fisheye_left_D=np.asarray(left["distortion"][...], dtype=np.float64),
        fisheye_left_size=(ls[0], ls[1]),
        fisheye_right_K=np.asarray(right["intrinsics"][...], dtype=np.float64),
        fisheye_right_D=np.asarray(right["distortion"][...], dtype=np.float64),
        fisheye_right_size=(rs[0], rs[1]),
        rectified_K=np.asarray(rect["intrinsics"][...], dtype=np.float64),
        rectified_size=(rect_s[0], rect_s[1]),
        T_right_left=np.asarray(g["stereo/T_right_left"][...], dtype=np.float64),
        T_head_camera=np.asarray(g["T_head_camera"][...], dtype=np.float64),
        T_mocapworld_slamworld=_load_mocapworld_slamworld(g),
        R_rect_left=R_left,
        R_rect_right=R_right,
        hamer_camera_side=hamer_side,
        left_half=int(left_half),
        rectification_source=source,
    )


def _load_task(
    f: h5py.File,
    release_dir: Path,
    external_json: Optional[Path],
) -> TaskDescription:
    """Load activity text from HDF5, sidecar JSON, or a placeholder."""
    if "behavior_annotation" in f:
        g = f["behavior_annotation"]
        if "activity_summarization" in g:
            ds = g["activity_summarization"]
            available = bool(ds.attrs.get("available", True))
            text = str(_decode(ds[()])).strip()
            if available and text:
                return TaskDescription(
                    activity_summarization=text,
                    source="annotation.hdf5:behavior_annotation/activity_summarization",
                )

    candidates = []
    if external_json is not None:
        candidates.append(Path(external_json))
    candidates.extend([release_dir / "task.json", release_dir / "caption.json"])

    for path in candidates:
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        text = str(
            data.get(
                "activity_summarization",
                data.get("main_task", data.get("Main Task", "")),
            )
        ).strip()
        if text:
            return TaskDescription(
                activity_summarization=text,
                source=str(path),
            )

    return TaskDescription(
        activity_summarization="(activity summarization not in release yet)",
        source="placeholder",
    )


def _format_scalar_for_list(val: Any) -> Any:
    """Format one HDF5 scalar for ``list_annotation_contents``."""
    if isinstance(val, (np.ndarray, np.generic)):
        if val.size == 0:
            return "[]"
        raw = val.flat[0]
        if isinstance(raw, (bytes, str, np.bytes_)):
            s = _decode(raw) if isinstance(raw, (bytes, np.bytes_)) else str(raw)
            return s[:60] + "..." if len(s) > 60 else s
        if np.issubdtype(getattr(val.dtype, "base", val.dtype), np.floating):
            return float(raw)
        if np.issubdtype(getattr(val.dtype, "base", val.dtype), np.integer):
            return int(raw)
        return str(raw)
    if isinstance(val, (bytes, np.bytes_)):
        return _decode(val)[:60]
    return val


def list_annotation_contents(annotation_path: str | Path) -> Dict[str, Any]:
    """List groups and datasets in ``annotation.hdf5``.

    Groups are labeled ``\"group\"``, arrays as their shape, scalars as values.
    """
    out: Dict[str, Any] = {}

    def _visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Group):
            out[name] = "group"
            return
        try:
            shape = obj.shape
            if shape == ():
                out[name] = _format_scalar_for_list(obj[()])
            else:
                out[name] = shape
        except Exception:
            out[name] = "?"

    with h5py.File(annotation_path, "r") as f:
        f.visititems(_visit)
    return out


def load_release_episode(
    release_dir: str | Path,
    annotation_name: str = "annotation.hdf5",
    task_json: Optional[str | Path] = None,
    load_depth: bool = True,
    load_point_cloud: bool = True,
    point_cloud_max_points: int = 80000,
    require_videos: bool = True,
) -> ReleaseEpisode:
    """Load one HumanPlus-1000 session.

    Args:
        release_dir: Session folder containing ``annotation.hdf5`` and videos.
        annotation_name: HDF5 filename inside ``release_dir``.
        task_json: Optional caption JSON if HDF5 has no behavior annotation.
        load_depth: If True, load ``depth/depth`` into memory.
        load_point_cloud: If True, load and optionally subsample SLAM points.
        point_cloud_max_points: Max SLAM map points to keep.
        require_videos: If True, require stereo mp4 files to exist.

    Returns:
        Populated ``ReleaseEpisode``.
    """
    root = Path(release_dir)
    ann_path = root / annotation_name
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation not found: {ann_path}")

    with h5py.File(ann_path, "r") as f:
        attrs = {k: _decode(v) for k, v in f.attrs.items()}
        unavailable: Tuple[str, ...] = ()
        if "metadata" in f and "unavailable_products" in f["metadata"].attrs:
            raw = _decode(f["metadata"].attrs["unavailable_products"])
            try:
                unavailable = tuple(json.loads(raw)) if isinstance(raw, str) else tuple(raw)
            except Exception:
                unavailable = (str(raw),)

        calib = _load_calibration(f, root)
        video_g = f["video"]
        left_name = _video_filename(video_g, "fisheye_left_file", "fisheye_left.mp4")
        right_name = _video_filename(video_g, "fisheye_right_file", "fisheye_right.mp4")

        video_timestamp_ns = np.asarray(video_g["timestamp_ns"][...], dtype=np.int64)
        n_video = int(video_timestamp_ns.shape[0])
        video_frame_indices = _int_dataset_or_arange(video_g, "frame_indices", n_video)

        body = f["body_motion"]
        hand = f["hand_motion"]
        sync = f["synchronization"]
        slam = f["slam"]
        depth_g = f["depth"] if "depth" in f else None

        sync_utc_ns = np.asarray(sync["utc_ns"][...], dtype=np.int64)
        n_motion = int(sync_utc_ns.shape[0])
        sync_video_frame_indices = np.asarray(
            sync["video_frame_indices"][...], dtype=np.int64
        )
        if "video_time_error_ns" in sync:
            sync_video_time_error_ns = np.asarray(
                sync["video_time_error_ns"][...], dtype=np.int64
            )
        else:
            sync_video_time_error_ns = np.zeros((n_motion,), dtype=np.int64)
        if "depth_frame_indices" in sync:
            sync_depth_frame_indices = np.asarray(
                sync["depth_frame_indices"][...], dtype=np.int64
            )
        else:
            sync_depth_frame_indices = sync_video_frame_indices.copy()

        slam_pc = np.zeros((0, 3), dtype=np.float32)
        if load_point_cloud and "point_cloud" in slam:
            slam_pc = np.asarray(slam["point_cloud"][...], dtype=np.float32)
            if slam_pc.shape[0] > point_cloud_max_points:
                rng = np.random.default_rng(0)
                idx = rng.choice(slam_pc.shape[0], size=point_cloud_max_points, replace=False)
                slam_pc = slam_pc[idx]

        slam_T_camera = np.asarray(slam["T_slamworld_camera"][...], dtype=np.float32)
        if "translation" in slam:
            slam_translation = np.asarray(slam["translation"][...], dtype=np.float32)
        else:
            slam_translation = slam_T_camera[:, :3, 3].astype(np.float32, copy=False)
        slam_quaternion = _optional_dataset(slam, "quaternion")
        if slam_quaternion is None:
            slam_quaternion = np.zeros((slam_T_camera.shape[0], 4), dtype=np.float32)
        slam_frame_indices = _int_dataset_or_arange(slam, "frame_indices", slam_T_camera.shape[0])
        if "utc_ns" in slam:
            slam_utc_ns = np.asarray(slam["utc_ns"][...], dtype=np.int64)
        else:
            slam_utc_ns = video_timestamp_ns

        depth = None
        depth_attrs: Dict[str, Any] = {}
        depth_frame_indices = np.zeros((0,), dtype=np.int64)
        depth_utc_ns = np.zeros((0,), dtype=np.int64)
        depth_K = None
        if depth_g is not None:
            depth_attrs = {k: _decode(v) for k, v in depth_g.attrs.items()}
            if "frame_indices" in depth_g:
                depth_frame_indices = np.asarray(depth_g["frame_indices"][...], dtype=np.int64)
            else:
                depth_frame_indices = sync_video_frame_indices.copy()
            if "utc_ns" in depth_g:
                depth_utc_ns = np.asarray(depth_g["utc_ns"][...], dtype=np.int64)
            else:
                depth_utc_ns = sync_utc_ns
            depth_K = np.asarray(depth_g["intrinsics"][...], dtype=np.float64)
            if load_depth and "depth" in depth_g:
                depth = np.asarray(depth_g["depth"][...], dtype=np.float32)

        if "foot_contact" in body:
            body_foot_contact = np.asarray(body["foot_contact"][...])
        elif "foot_contact_probability" in body:
            body_foot_contact = np.asarray(
                body["foot_contact_probability"][...], dtype=np.float32
            ) >= 0.5
        else:
            body_foot_contact = np.zeros((n_motion, 2), dtype=bool)

        if "frame_indices" in body:
            body_frame_indices = np.asarray(body["frame_indices"][...], dtype=np.int64)
        else:
            body_frame_indices = sync_video_frame_indices.copy()
        if "utc_ns" in body:
            body_utc_ns = np.asarray(body["utc_ns"][...], dtype=np.int64)
        else:
            body_utc_ns = sync_utc_ns

        def _hand_T_wrist(side: str) -> np.ndarray:
            path = f"{side}/T_mocapworld_wrist"
            loaded = _optional_dataset(hand, path)
            if loaded is not None:
                return loaded.astype(np.float32, copy=False)
            return np.zeros((n_motion, 4, 4), dtype=np.float32)

        task = _load_task(f, root, Path(task_json) if task_json else None)

        episode = ReleaseEpisode(
            root=root,
            attrs=attrs,
            calib=calib,
            video_left_path=root / left_name,
            video_right_path=root / right_name,
            video_timestamp_ns=video_timestamp_ns,
            video_frame_indices=video_frame_indices,
            sync_utc_ns=sync_utc_ns,
            sync_video_frame_indices=sync_video_frame_indices,
            sync_video_time_error_ns=sync_video_time_error_ns,
            sync_depth_frame_indices=sync_depth_frame_indices,
            body_keypoints=np.asarray(body["body_keypoints"][...], dtype=np.float32),
            body_T_root=np.asarray(body["T_mocapworld_root"][...], dtype=np.float32),
            body_smplh_pose=np.asarray(body["smplh_pose"][...], dtype=np.float32),
            body_foot_contact=body_foot_contact,
            body_frame_indices=body_frame_indices,
            body_utc_ns=body_utc_ns,
            hand_left_T_wrist=_hand_T_wrist("left"),
            hand_right_T_wrist=_hand_T_wrist("right"),
            hand_left_pose=np.asarray(hand["left/mano_hand_pose"][...], dtype=np.float32),
            hand_right_pose=np.asarray(hand["right/mano_hand_pose"][...], dtype=np.float32),
            hand_left_valid=np.asarray(hand["left/valid"][...]).astype(bool),
            hand_right_valid=np.asarray(hand["right/valid"][...]).astype(bool),
            hand_left_confidence=np.asarray(hand["left/confidence"][...], dtype=np.float32),
            hand_right_confidence=np.asarray(
                hand["right/confidence"][...], dtype=np.float32
            ),
            hand_left_joints_3d=_optional_dataset(hand, "left/joints_3d"),
            hand_right_joints_3d=_optional_dataset(hand, "right/joints_3d"),
            hand_left_wrist_position_camera=_optional_dataset(
                hand, "left/wrist_position_camera"
            ),
            hand_right_wrist_position_camera=_optional_dataset(
                hand, "right/wrist_position_camera"
            ),
            slam_T_camera=slam_T_camera,
            slam_translation=slam_translation,
            slam_quaternion=np.asarray(slam_quaternion, dtype=np.float32),
            slam_point_cloud=slam_pc,
            slam_frame_indices=slam_frame_indices,
            slam_utc_ns=slam_utc_ns,
            depth=depth,
            depth_frame_indices=depth_frame_indices,
            depth_utc_ns=depth_utc_ns,
            depth_intrinsics=depth_K,
            depth_attrs=depth_attrs,
            task=task,
            unavailable_products=unavailable,
        )

    if (
        episode.hand_left_wrist_position_camera is None
        and episode.hand_right_wrist_position_camera is None
    ):
        left_w, right_w = _load_wrist_position_camera_from_hamer_jsonl(
            root, episode.sync_video_frame_indices
        )
        episode.hand_left_wrist_position_camera = left_w
        episode.hand_right_wrist_position_camera = right_w

    if require_videos:
        for path in (episode.video_left_path, episode.video_right_path):
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}")
    return episode


load_session = load_release_episode


class StereoVideoReader:
    """Read left/right fisheye videos in lockstep by frame index."""

    def __init__(self, left_path: Path, right_path: Path):
        self.left = cv2.VideoCapture(str(left_path))
        self.right = cv2.VideoCapture(str(right_path))
        if not self.left.isOpened() or not self.right.isOpened():
            raise RuntimeError(f"Failed to open video: {left_path} / {right_path}")
        self._cursor = -1

    def read_index(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read one stereo pair as BGR ``(left, right)``."""
        if index < 0:
            raise IndexError(index)
        if index != self._cursor + 1:
            self.left.set(cv2.CAP_PROP_POS_FRAMES, float(index))
            self.right.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok_l, frame_l = self.left.read()
        ok_r, frame_r = self.right.read()
        if not ok_l or not ok_r:
            raise RuntimeError(f"Failed to read video frame: index={index}")
        self._cursor = index
        return frame_l, frame_r

    def close(self) -> None:
        """Release OpenCV capture handles."""
        self.left.release()
        self.right.release()

    def __enter__(self) -> "StereoVideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def build_video_to_depth_index(
    depth_frame_indices: np.ndarray,
    num_video_frames: int,
) -> np.ndarray:
    """Map ``video_frame -> depth_array_index`` (``-1`` if unmatched)."""
    mapping = np.full((num_video_frames,), -1, dtype=np.int64)
    for depth_i, vid_i in enumerate(np.asarray(depth_frame_indices).tolist()):
        vi = int(vid_i)
        if 0 <= vi < num_video_frames:
            mapping[vi] = int(depth_i)
    return mapping


def format_task_markdown(task: TaskDescription) -> str:
    """Markdown for the Behavior Annotation panel."""
    return (
        "### Behavior Annotation\n\n"
        f"**Activity Summarization**  \n{task.activity_summarization}\n"
    )
