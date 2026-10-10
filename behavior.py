"""Behavior documents and a native action timeline on the video clock."""
from __future__ import annotations

import re

import numpy as np
import rerun as rr

from .constants import PATH_ACTION_TIMELINE, PATH_CURRENT_ACTION
from .data_loader import AtomicAction, ReleaseEpisode

# Bright, distinct colors on both light and dark Rerun themes.
ACTION_COLORS = (
    (230, 75, 70), (40, 170, 90), (55, 130, 235), (210, 155, 30),
    (155, 90, 220), (25, 175, 185), (220, 90, 160), (225, 120, 40),
    (115, 165, 45), (80, 100, 200), (185, 95, 65), (85, 160, 145),
)


def video_seconds(episode: ReleaseEpisode) -> np.ndarray:
    """Subtract integer nanoseconds before converting to float (avoid UTC loss)."""
    timestamps = episode.video_timestamp_ns
    if not len(timestamps):
        raise ValueError("Video timestamps are required for action synchronization")
    times = (timestamps - timestamps[0]).astype(np.float64) / 1e9
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Video timestamps must be finite and strictly increasing")
    return times


def active_actions(actions: tuple[AtomicAction, ...], seconds: float) -> tuple[int, ...]:
    return tuple(i for i, a in enumerate(actions) if a.start_s <= seconds < a.end_s)


def _text(value: str) -> str:
    return re.sub(r'([\\`*_{}\[\]<>#|])', r'\\\1', value).replace('\n', ' ')


def action_context(actions: tuple[AtomicAction, ...], seconds: float):
    active = active_actions(actions, seconds)
    if active:
        return active, min(active) - 1, max(active) + 1
    ended = [i for i, a in enumerate(actions) if a.end_s <= seconds]
    future = [i for i, a in enumerate(actions) if a.start_s > seconds]
    previous = max(ended, key=lambda i: actions[i].end_s) if ended else -1
    following = min(future, key=lambda i: actions[i].start_s) if future else len(actions)
    return active, previous, following


def format_current_action(actions: tuple[AtomicAction, ...], seconds: float) -> str:
    if not actions:
        return "**No action annotations**  \nThis session contains no available atomic-action labels."
    active, previous, following = action_context(actions, seconds)
    chunks = []
    if active:
        if len(active) > 1:
            chunks.append("**Overlapping actions**")
        for i in active:
            a = actions[i]
            chunks.append(
                f"**{_text(a.label)}**  \n"
                f"{a.start_s:.2f}–{a.end_s:.2f} s · Action {i + 1} / {len(actions)}"
            )
    else:
        chunks.append("**No action label at this time**")
    prev = _text(actions[previous].label) if previous >= 0 else "—"
    nxt = _text(actions[following].label) if following < len(actions) else "—"
    chunks.append(f"Previous: {prev}\u2003\u2003Next: {nxt}")
    return "\n\n".join(chunks)


def timeline_segments(actions: tuple[AtomicAction, ...], duration: float):
    """Independent line intervals avoid connecting repeated actions across gaps.

    Most sessions use one shared baseline. Overlapping intervals get separate
    lanes so that neither concurrent annotation is hidden by the other.
    """
    lane_ends: list[float] = []
    segments = []
    for i, a in enumerate(actions):
        start, end = max(0.0, a.start_s), min(duration, a.end_s)
        if end <= start:
            continue
        lane = next((j for j, last in enumerate(lane_ends) if last <= start), len(lane_ends))
        if lane == len(lane_ends):
            lane_ends.append(end)
        else:
            lane_ends[lane] = end
        segments.append((i, start, end, lane))
    return [(i, start, end, lane * .8 / max(1, len(lane_ends) - 1))
            for i, start, end, lane in segments]


class BehaviorDisplay:
    def __init__(self, episode: ReleaseEpisode):
        self.actions = episode.atomic_actions
        self.times = video_seconds(episode)
        interval = float(np.median(np.diff(self.times))) if len(self.times) > 1 else 1 / 30
        self.duration = max(interval, float(self.times[-1]) + interval)
        self.segments = timeline_segments(self.actions, self.duration)
        self.boundaries = sorted({0.0} | {t for _, start, end, _ in self.segments for t in (start, end)})
        self._boundary_cursor = 0
        self._last_seconds = -float('inf')

    def log_static(self):
        """Only styles are static; never publish future timestamped samples here."""
        rr.reset_time()
        label_colors = {}
        for i, _, _, _ in self.segments:
            action = self.actions[i]
            color = label_colors.setdefault(action.label, ACTION_COLORS[len(label_colors) % len(ACTION_COLORS)])
            rr.log(f'{PATH_ACTION_TIMELINE}/action_{i + 1:05d}', rr.SeriesLines(
                colors=color, widths=3, names=f'{i + 1:02d} · {action.label}',
            ), static=True)

    def log_document(self, seconds: float):
        rr.log(PATH_CURRENT_ACTION, rr.TextDocument(format_current_action(self.actions, seconds), media_type=rr.MediaType.MARKDOWN))

    def _log_at(self, seconds: float):
        rr.set_time('stable_time', duration=seconds)
        rr.set_time('frame', sequence=min(len(self.times) - 1, int(np.searchsorted(self.times, seconds, side='left'))))
        self.log_document(seconds)
        for i, start, end, lane in self.segments:
            # Include the endpoint to finish the line, while documents use [start, end).
            if start <= seconds <= end:
                rr.log(f'{PATH_ACTION_TIMELINE}/action_{i + 1:05d}', rr.Scalars(lane))

    def advance(self, seconds: float):
        """Publish exact action boundaries before this frame, never beyond it.

        Initial live-follow and later replay share the same chronological data.
        Reverse seeking in Rerun reads recorded documents, not this producer.
        """
        if seconds < self._last_seconds:
            raise ValueError('Video frames must be logged in chronological order')
        while self._boundary_cursor < len(self.boundaries):
            boundary = self.boundaries[self._boundary_cursor]
            if boundary > seconds:
                break
            self._log_at(boundary)
            self._boundary_cursor += 1
        self._log_at(seconds)
        self._last_seconds = seconds
