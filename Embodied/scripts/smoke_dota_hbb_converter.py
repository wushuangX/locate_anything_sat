#!/usr/bin/env python
"""Smoke test for DOTA HBB conversion with a tiny synthetic DOTA layout."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def write_tiny_dota(root: Path, with_second_image: bool = True) -> None:
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
    if with_second_image:
        # One ship fully inside the left tile, so 9:1 over 2 images clamps to
        # exactly 1 internal-val image with at least one kept box per partition.
        Image.new("RGB", (672, 448), color=(65, 43, 21)).save(image_dir / "P0002.png")
        (label_dir / "P0002.txt").write_text(
            "imagesource:synthetic\n"
            "gsd:1.0\n"
            "10 20 60 20 60 70 10 70 ship 0\n",
            encoding="utf-8",
        )


def run_converter(converter: Path, dota_root: Path, output_root: Path, extra: list[str]) -> None:
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
            *extra,
        ],
        check=True,
    )


def check_task_mix_run(tmp_path: Path, converter: Path) -> None:
    dota_root = tmp_path / "DOTA-v1.0"
    output_root = tmp_path / "converted_mix"
    write_tiny_dota(dota_root, with_second_image=True)
    run_converter(converter, dota_root, output_root, ["--samples-per-tile", "3", "--recipe-name", "tiny_dota"])

    ann_dir = output_root / "annotations"
    train_lines = [
        json.loads(line)
        for line in (ann_dir / "DOTA-v1.0_train_mix_hbb_448.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    internal_val_lines = [
        json.loads(line)
        for line in (ann_dir / "DOTA-v1.0_internal_val_mix_hbb_448.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(train_lines) >= 1, train_lines
    assert len(internal_val_lines) >= 1, internal_val_lines
    for line in train_lines + internal_val_lines:
        assert len(line["conversations"]) == 2, line
        assert line["image"].startswith("tiles/train/"), line
    answers = [line["conversations"][1]["value"] for line in train_lines + internal_val_lines]
    assert any(re.search(r"<ref>.+</ref><box><", answer) for answer in answers), answers
    assert (output_root / "tiles" / "train" / "P0001__x0_y0_s448.png").exists()
    assert (output_root / "tiles" / "train" / "P0001__x224_y0_s448.png").exists()

    train_only = json.loads((output_root / "recipes" / "tiny_dota_train_only.json").read_text(encoding="utf-8"))
    assert len(train_only) == 1, train_only
    assert list(train_only)[0].endswith("_train_mix_hbb_448"), train_only
    combined = json.loads((output_root / "recipes" / "tiny_dota.json").read_text(encoding="utf-8"))
    assert sorted(combined) == ["tiny_dota_internal_val_mix_hbb_448", "tiny_dota_train_mix_hbb_448"], combined
    metadata = json.loads((output_root / "metadata" / "tiny_dota_stats.json").read_text(encoding="utf-8"))
    assert metadata["task_mix"] is True, metadata
    assert "task_mix_weights" in metadata, metadata


def check_helpers(converter: Path) -> None:
    spec = importlib.util.spec_from_file_location("convert_dota_hbb_to_locany_smoke", converter)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.guaranteed_absent({"ship", "plane"}, {"ship"}) == {"plane"}
    negative = module.make_negative_sample("t.png", "ship")
    assert negative["conversations"][0]["value"] == "Locate all the instances that matches the following description: ship.", negative
    assert negative["conversations"][1]["value"] == "<box>none</box>", negative
    referring = module.make_referring_sample("t.png", "the leftmost ship", (10, 20, 30, 40))
    assert referring["conversations"][0]["value"] == (
        "Locate a single instance that matches the following description: the leftmost ship."
    ), referring
    gpt = referring["conversations"][1]["value"]
    assert "<ref>the leftmost ship</ref><box><10><20><30><40></box>" in gpt, gpt


def check_legacy_run(tmp_path: Path, converter: Path) -> None:
    dota_root = tmp_path / "legacy" / "DOTA-v1.0"
    output_root = tmp_path / "converted_legacy"
    write_tiny_dota(dota_root, with_second_image=False)
    run_converter(converter, dota_root, output_root, ["--no-task-mix", "--max-boxes-per-sample", "1", "--recipe-name", "tiny_dota"])
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


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    converter = repo_root / "scripts" / "convert_dota_hbb_to_locany.py"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        check_task_mix_run(tmp_path, converter)
        check_helpers(converter)
        check_legacy_run(tmp_path, converter)
    print("dota hbb converter smoke passed")


if __name__ == "__main__":
    main()
