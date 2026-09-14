"""SMPL-H forward kinematics (numpy LBS) for ``body_motion/smplh_pose``.

Uses the official ``model.npz`` (``v_template``, ``shapedirs``, ``posedirs``,
``J_regressor``, ``weights``, ``kintree_table``, ``f``). No torch / smplx.

Pose convention: ``use_pca=False``, ``flat_hand_mean=True`` — 156 axis-angle
values (22 body + 15 left hand + 15 right hand) as local rotations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SMPLH_NUM_JOINTS = 52
SMPLH_POSE_DIM = 156


def _model_search_paths(gender: str) -> List[Path]:
    """Candidate paths for SMPL-H ``model.npz``, highest priority first."""
    candidates: List[Path] = []
    env = os.environ.get("HUMANPLUS_SMPLH_MODEL") or os.environ.get("EGOGARMENT_SMPLH_MODEL")
    if env:
        candidates.append(Path(env))
    repo_root = Path(__file__).resolve().parent.parent
    candidates += [
        repo_root / "third_party" / "smplh" / gender / "model.npz",
        repo_root.parent / "smplx" / "smplh" / gender / "model.npz",
        Path.home() / "Desktop" / "smplx" / "smplh" / gender / "model.npz",
    ]
    return candidates


def resolve_smplh_model(
    model_path: Optional[str | Path] = None,
    gender: str = "male",
) -> Optional[Path]:
    """Resolve an SMPL-H ``model.npz`` path, or ``None`` if missing."""
    if model_path is not None:
        p = Path(model_path).expanduser()
        return p if p.is_file() else None
    for cand in _model_search_paths(gender):
        if cand.is_file():
            return cand
    return None


def _axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Convert ``(J, 3)`` axis-angles to ``(J, 3, 3)`` rotation matrices."""
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    safe = np.where(theta < 1e-8, 1.0, theta)
    k = aa / safe
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    zero = np.zeros_like(kx)
    K = np.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1
    ).reshape(-1, 3, 3)
    s = np.sin(theta)[:, :, None]
    c = np.cos(theta)[:, :, None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    R = eye + s * K + (1.0 - c) * (K @ K)
    return np.where((theta < 1e-8)[:, :, None], eye, R)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Accumulate face normals onto vertices."""
    tris = vertices[faces]
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    out = np.zeros_like(vertices)
    n_v = vertices.shape[0]
    for axis in range(3):
        out[:, axis] = np.bincount(
            faces.reshape(-1), weights=np.repeat(face_n[:, axis], 3), minlength=n_v
        )
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-8)


@dataclass
class SmplhForward:
    """One SMPL-H forward result."""

    vertices: np.ndarray
    normals: np.ndarray
    joints: np.ndarray


class SmplhModel:
    """SMPL-H LBS with zero betas."""

    def __init__(self, model_path: str | Path, num_betas: int = 16) -> None:
        data = np.load(str(model_path), allow_pickle=True)
        self.faces = np.asarray(data["f"], dtype=np.uint32)
        self.v_template = np.asarray(data["v_template"], dtype=np.float64)
        self.shapedirs = np.asarray(data["shapedirs"], dtype=np.float64)[
            :, :, :num_betas
        ]
        self.posedirs = np.asarray(data["posedirs"], dtype=np.float64).reshape(-1, 459)
        self.J_regressor = np.asarray(data["J_regressor"], dtype=np.float64)
        self.weights = np.asarray(data["weights"], dtype=np.float64)
        parents = np.asarray(data["kintree_table"], dtype=np.int64)[0].copy()
        parents[0] = -1
        self.parents = parents
        self.num_joints = int(self.weights.shape[1])

        self.v_shaped = self.v_template
        self.J_rest = self.J_regressor @ self.v_shaped

    def forward(self, pose_aa: np.ndarray) -> SmplhForward:
        """Map ``(156,)`` or ``(52, 3)`` axis-angles to vertices / normals / joints."""
        pose = np.asarray(pose_aa, dtype=np.float64).reshape(-1, 3)
        if pose.shape[0] < self.num_joints:
            pose = np.vstack(
                [pose, np.zeros((self.num_joints - pose.shape[0], 3))]
            )
        R = _axis_angle_to_matrix(pose[: self.num_joints])

        pose_feature = (R[1:] - np.eye(3)).reshape(-1)
        v_posed = self.v_shaped + (self.posedirs @ pose_feature).reshape(-1, 3)

        G = np.zeros((self.num_joints, 4, 4), dtype=np.float64)
        G[0, :3, :3] = R[0]
        G[0, :3, 3] = self.J_rest[0]
        G[0, 3, 3] = 1.0
        for j in range(1, self.num_joints):
            local = np.eye(4)
            local[:3, :3] = R[j]
            local[:3, 3] = self.J_rest[j] - self.J_rest[self.parents[j]]
            G[j] = G[self.parents[j]] @ local

        joints = G[:, :3, 3].copy()
        A = G.copy()
        A[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], self.J_rest)

        T = (self.weights @ A.reshape(self.num_joints, 16)).reshape(-1, 4, 4)
        v_h = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
        vertices = np.einsum("vab,vb->va", T, v_h)[:, :3]

        return SmplhForward(
            vertices=vertices.astype(np.float32),
            normals=_vertex_normals(vertices, self.faces.astype(np.int64)).astype(
                np.float32
            ),
            joints=joints.astype(np.float32),
        )

    def forward_at_pelvis(
        self,
        pose_aa: np.ndarray,
        pelvis_xyz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward kinematics, then translate so the pelvis is at ``pelvis_xyz``.

        Aligns the mesh with ``body_keypoints[0]`` without using ``T_mocapworld_root``.
        """
        out = self.forward(pose_aa)
        offset = np.asarray(pelvis_xyz, dtype=np.float32).reshape(3) - out.joints[0]
        return out.vertices + offset, out.normals
