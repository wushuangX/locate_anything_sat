#!/usr/bin/env python
"""Smoke test for the DOTA geometric JSONL augmentation path.

Covers: transform_xy identities, transform_box math, the JSONL rewriter
(T5 drop + rot90 rewrite), apply_image_geometry, apply_color_jitter.
Run from repo root or Embodied/: python scripts/smoke_augment_locany_jsonl.py
"""

from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_scripts_dir = str(Path(__file__).resolve().parent)
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from PIL import Image

from eaglevl.train.augmentation import (
    apply_color_jitter,
    apply_image_geometry,
    transform_box,
    transform_xy,
)
from augment_locany_jsonl import main as rewriter_main


def check_transform_identities() -> None:
    points = [(0, 0), (1000, 1000), (137, 963), (100, 200)]
    for x, y in points:
        cx, cy = x, y
        for _ in range(4):
            cx, cy = transform_xy(cx, cy, "rot90")
        assert (cx, cy) == (x, y), f"rot90 x4 not identity at {(x, y)}: {(cx, cy)}"

        cx, cy = transform_xy(x, y, "rot180")
        cx, cy = transform_xy(cx, cy, "rot180")
        assert (cx, cy) == (x, y), f"rot180 x2 not identity at {(x, y)}: {(cx, cy)}"

        cx, cy = transform_xy(x, y, "hflip")
        cx, cy = transform_xy(cx, cy, "hflip")
        assert (cx, cy) == (x, y), f"hflip x2 not identity at {(x, y)}: {(cx, cy)}"

    assert transform_box(100, 200, 300, 400, "rot90") == (200, 700, 400, 900), (
        f"ship box rot90: {transform_box(100, 200, 300, 400, 'rot90')}"
    )
    assert transform_box(500, 100, 700, 250, "rot90") == (100, 300, 250, 500), (
        f"plane box rot90: {transform_box(500, 100, 700, 250, 'rot90')}"
    )
    print("[smoke] transform identities + box math OK")


def check_rewriter() -> None:
    t1 = {
        "conversations": [
            {
                "from": "human",
                "value": "Locate all the instances that matches the following description: ship</c>plane.",
            },
            {
                "from": "gpt",
                "value": (
                    "<ref>ship</ref><box><100><200><300><400></box>"
                    "<ref>plane</ref><box><500><100><700><250></box>"
                ),
            },
        ],
        "image": "tiles/train/__smoke__.png",
    }
    t5 = {
        "conversations": [
            {
                "from": "human",
                "value": "Locate a single instance that matches the following description: the leftmost ship.",
            },
            {
                "from": "gpt",
                "value": "<ref>the leftmost ship</ref><box><100><200><300><400></box>",
            },
        ],
        "image": "tiles/train/__smoke__.png",
    }
    with tempfile.TemporaryDirectory() as tmp:
        inp = Path(tmp) / "in.jsonl"
        inp.write_text(
            "\n".join(json.dumps(s) for s in (t1, t5)) + "\n", encoding="utf-8"
        )
        rewriter_main(
            ["--input", str(inp), "--output-dir", tmp, "--stem", "smoke", "--ops", "rot90"]
        )
        out = Path(tmp) / "smoke_rot90.jsonl"
        lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1, f"expected 1 line (T5 dropped), got {len(lines)}"
        gpt = lines[0]["conversations"][1]["value"]
        assert "<ref>ship</ref><box><200><700><400><900></box>" in gpt, gpt
        assert "<ref>plane</ref><box><100><300><250><500></box>" in gpt, gpt
        assert lines[0]["image"] == "tiles/train/__smoke__.png", "image path must be untouched"
        assert lines[0]["conversations"][0]["value"] == t1["conversations"][0]["value"]
    print("[smoke] rewriter: T5 dropped, rot90 boxes rewritten, image path kept OK")


def check_image_ops() -> None:
    img = Image.new("RGB", (448, 448))
    for x in range(0, 448, 64):
        for y in range(0, 448, 64):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))

    assert apply_image_geometry(img, 0, False) is img, "no-op must return same object"
    assert apply_image_geometry(img, 90, False).size == img.size == (448, 448)
    assert apply_image_geometry(img, 180, True).size == (448, 448)
    try:
        apply_image_geometry(img, 45, False)
        raise AssertionError("rotate=45 must raise ValueError")
    except ValueError:
        pass

    assert apply_color_jitter(img, enabled=False) is img, "disabled jitter must return same object"
    random.seed(0)
    jittered = apply_color_jitter(img, enabled=True, strength=0.2)
    assert isinstance(jittered, Image.Image), type(jittered)
    assert jittered.size == img.size, jittered.size
    print("[smoke] apply_image_geometry / apply_color_jitter OK")


def main() -> None:
    check_transform_identities()
    check_rewriter()
    check_image_ops()
    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
