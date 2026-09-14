from __future__ import annotations

import argparse
from pathlib import Path

from yolo_motion.source_probe import materialize_clips


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize benchmark MP4 clips from downloaded public sources."
    )
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--clips", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in materialize_clips(
        args.sources,
        args.clips,
        args.source_dir,
        args.output_dir,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
