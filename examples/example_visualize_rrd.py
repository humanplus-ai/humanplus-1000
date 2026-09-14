"""Example: visualize a HumanPlus-1000 session with Rerun.

Run from the toolkit root:

    python examples/example_visualize_rrd.py --data_root /path/to/session --output_rrd vis.rrd
"""

from __future__ import annotations

import argparse

from _bootstrap import setup

setup()

from humanplus_viewer.visualize import visualize_release  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Log a HumanPlus-1000 session to Rerun / an .rrd file."
    )
    parser.add_argument(
        "--data_root",
        "--session-dir",
        dest="data_root",
        type=str,
        required=True,
        help="Session folder (annotation.hdf5 + fisheye mp4)",
    )
    parser.add_argument(
        "--output_rrd",
        "--output-rrd",
        dest="output_rrd",
        type=str,
        default=None,
        help="Optional path to save a .rrd recording",
    )
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--no-spawn", action="store_true")
    parser.add_argument("--smplh-model", type=str, default=None)
    args = parser.parse_args()

    visualize_release(
        release_dir=args.data_root,
        output_rrd=args.output_rrd,
        spawn=not args.no_spawn,
        max_frames=args.max_frames,
        stride=args.stride,
        image_scale=args.image_scale,
        smplh_model_path=args.smplh_model,
    )
    if args.output_rrd:
        print(f"Wrote {args.output_rrd}. Open with: rerun {args.output_rrd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
