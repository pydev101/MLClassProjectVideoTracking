"""Regenerate ``combined.json`` from Part A, Part B, and Mall train/test lists."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent

    def load(name: str) -> list:
        with open(root / name, encoding="utf-8") as f:
            return json.load(f)

    train = (
        load("part_A_train.json")
        + load("part_B_train.json")
        + load("mall_train.json")
    )
    test = (
        load("part_A_test.json")
        + load("part_B_test.json")
        + load("mall_test.json")
    )
    out_path = root / "combined.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"train": train, "test": test}, f, indent=2)
    print(f"Wrote {out_path} ({len(train)} train, {len(test)} test images)")


if __name__ == "__main__":
    main()
