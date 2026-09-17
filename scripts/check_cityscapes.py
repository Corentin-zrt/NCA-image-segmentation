from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nca_segmentation.v2.data.cityscapes import _find_dataset_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a Cityscapes directory before training")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for name in ("gtFine", "leftImg8bit"):
        try:
            folder = _find_dataset_folder(args.root, name)
        except FileNotFoundError as error:
            print(f"MISSING {name}: {error}")
            continue
        files = list(folder.glob("**/*.png"))
        print(f"OK {name}: {folder} ({len(files)} PNG files)")


if __name__ == "__main__":
    main()