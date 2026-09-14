"""Example: visualize a HumanPlus-1000 session in Rerun.

Run from the toolkit root:

    python examples/example_visualize.py --session-dir /path/to/session
"""

from __future__ import annotations

import argparse

from _bootstrap import setup

setup()

from humanplus_viewer.visualize import visualize_release  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open a HumanPlus-1000 session in the Rerun viewer."
    )
    parser.add_argument(
        "--session-dir",
        "--data_root",
        dest="session_dir",
        type=str,
        required=True,
        help="Session folder (annotation.hdf5 + fisheye mp4)",
    )
    parser.add_argument(
        "--output-rrd",
        "--output_rrd",
        dest="output_rrd",
        type=str,
        default=None,
        help="Optional Rerun recording file (.rrd)",
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--smplh-model", type=str, default=None)
    args = parser.parse_args()

    visualize_release(
        release_dir=args.session_dir,
        output_rrd=args.output_rrd,
        spawn=not args.no_spawn,
        max_frames=args.max_frames,
        stride=args.stride,
        image_scale=args.image_scale,
        smplh_model_path=args.smplh_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
