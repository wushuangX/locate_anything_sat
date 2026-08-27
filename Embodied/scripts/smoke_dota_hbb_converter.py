#!/usr/bin/env python
"""Smoke test for DOTA HBB conversion with a tiny synthetic DOTA layout."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def write_tiny_dota(root: Path) -> None:
    split = root / "train"
    image_dir = split / "images"
    label_dir = split / "labelTxt-v1.0"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (672, 448), color=(12, 34, 56)).save(image_dir / "P0001.png")
    (label_dir / "P0001.txt").write_text(
        "imagesource:synthetic\n"
        "gsd:1.0\n"
        "10 20 110 20 110 120 10 120 small-vehicle 0\n"
        "520 20 620 20 620 120 520 120 large-vehicle 1\n",
        encoding="utf-8",
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    converter = repo_root / "scripts" / "convert_dota_hbb_to_locany.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        dota_root = tmp_path / "DOTA-v1.0"
        output_root = tmp_path / "converted"
        write_tiny_dota(dota_root)
        subprocess.run(
            [
                sys.executable,
                str(converter),
                "--dota-root",
                str(dota_root),
                "--output-root",
                str(output_root),
                "--version",
                "v1.0",
                "--splits",
                "train",
                "--tile-size",
                "448",
                "--max-boxes-per-sample",
                "1",
                "--recipe-name",
                "tiny_dota",
            ],
            check=True,
        )
        jsonl_path = output_root / "annotations" / "DOTA-v1.0_train_hbb_448.jsonl"
        lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 2, lines
        answers = [line["conversations"][1]["value"] for line in lines]
        assert any("<ref>small-vehicle</ref>" in answer for answer in answers), answers
        assert any("<ref>large-vehicle</ref>" in answer for answer in answers), answers
        assert (output_root / "tiles" / "train" / "P0001__x0_y0_s448.png").exists()
        assert (output_root / "tiles" / "train" / "P0001__x224_y0_s448.png").exists()
        recipe = json.loads((output_root / "recipes" / "tiny_dota.json").read_text(encoding="utf-8"))
        assert "tiny_dota_train_hbb_448" in recipe
        train_only = json.loads((output_root / "recipes" / "tiny_dota_train_only.json").read_text(encoding="utf-8"))
        assert list(train_only) == ["tiny_dota_train_hbb_448"], train_only
    print("dota hbb converter smoke passed")


if __name__ == "__main__":
    main()
