#!/usr/bin/env python
"""Convert DOTA OBB annotations to tiled LocateAnything HBB JSONL.

The converter supports extracted DOTA-v1.0, DOTA-v1.5 and DOTA-v2.0 layouts.
It also supports v1.x image/label directories whose files are still inside zip
archives by reading from zip and writing 448x448 training tiles by default.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # DOTA-v2.0 has images up to ~800M pixels

PROMPT_TEMPLATE = "Locate all the instances that matches the following description: {classes}."
SINGLE_PROMPT_TEMPLATE = "Locate a single instance that matches the following description: {phrase}."
DOTA_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
TASK_MIX_WEIGHTS = {"t1": 0.50, "t2": 0.15, "t3": 0.10, "t4": 0.15, "t5": 0.10}
T4_SCENE_SCALE_SKIP = frozenset({"harbor", "bridge", "roundabout"})
T5_PREDICATES = ("leftmost", "rightmost", "topmost", "bottommost")


@dataclass(frozen=True)
class DotaObject:
    label: str
    difficult: int
    hbb: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    image_path: Path | None
    image_zip: Path | None
    image_member: str | None
    label_path: Path | None
    label_zip: Path | None
    label_member: str | None


@dataclass
class SplitStats:
    split: str
    images_seen: int = 0
    images_missing: int = 0
    images_without_objects: int = 0
    source_objects: int = 0
    kept_objects: int = 0
    tiles_written: int = 0
    samples_written: int = 0
    skipped_difficult_objects: int = 0
    skipped_small_or_invisible_objects: int = 0
    class_counts: Counter | None = None
    task_counts: Counter | None = None
    tiles_missing: int = 0
    n_internal_val_images: int = 0
    n_train_images: int = 0
    rl_single_samples_written: int = 0
    rl_single_max_boxes: int = 0
    rl_single_class_counts: Counter | None = None

    def __post_init__(self) -> None:
        if self.class_counts is None:
            self.class_counts = Counter()
        if self.task_counts is None:
            self.task_counts = Counter()
        if self.rl_single_class_counts is None:
            self.rl_single_class_counts = Counter()

    def to_dict(self) -> dict:
        return {
            "split": self.split,
            "images_seen": self.images_seen,
            "images_missing": self.images_missing,
            "images_without_objects": self.images_without_objects,
            "source_objects": self.source_objects,
            "kept_objects": self.kept_objects,
            "tiles_written": self.tiles_written,
            "samples_written": self.samples_written,
            "skipped_difficult_objects": self.skipped_difficult_objects,
            "skipped_small_or_invisible_objects": self.skipped_small_or_invisible_objects,
            "class_counts": dict(sorted(self.class_counts.items())),
            "task_counts": dict(sorted(self.task_counts.items())),
            "tiles_missing": self.tiles_missing,
            "n_internal_val_images": self.n_internal_val_images,
            "n_train_images": self.n_train_images,
            "rl_single_samples_written": self.rl_single_samples_written,
            "rl_single_max_boxes": self.rl_single_max_boxes,
            "rl_single_class_counts": dict(sorted(self.rl_single_class_counts.items())),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dota-root", type=Path, required=True, help="DOTA dataset root, e.g. /data/.../DOTA-v1.0")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for tiles, JSONL, recipe and metadata")
    parser.add_argument("--version", choices=["auto", "v1.0", "v1.5", "v2.0"], default="auto")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Labelled splits to convert, e.g. train val")
    parser.add_argument("--label-version", choices=["auto", "v1.0", "v1.5", "v2.0"], default="auto")
    parser.add_argument("--tile-size", type=int, default=448)
    parser.add_argument("--overlap", type=float, default=0.0, help="Tile overlap ratio in [0, 0.9)")
    parser.add_argument("--min-visibility", type=float, default=0.5, help="Min clipped HBB area/original HBB area")
    parser.add_argument("--min-box-size", type=float, default=2.0, help="Min clipped width/height in pixels")
    parser.add_argument("--max-boxes-per-sample", type=int, default=30)
    parser.add_argument("--include-difficult", action="store_true", default=True)
    parser.add_argument("--exclude-difficult", action="store_false", dest="include_difficult")
    parser.add_argument("--include-empty", action="store_true", help="Write <box>none</box> samples for empty tiles")
    parser.add_argument("--task-mix", action="store_true", default=True, help="Emit a static mixed-task JSONL: T1 all-classes detection, T2 single-class, T3 class subset, T4 pure negative, T5 referring (default on)")
    parser.add_argument("--no-task-mix", action="store_false", dest="task_mix", help="Restore the legacy one all-classes sample per box chunk")
    parser.add_argument("--samples-per-tile", type=int, default=6, help="Task-mix draws per non-empty tile (ignored with --no-task-mix)")
    parser.add_argument("--emit-rl-single-class", action="store_true",
                        help="Also emit full single-class GT rows for RL (train split, non-empty tiles): one row per (tile, class), never chunked, no T4/T5 mixing")
    parser.add_argument("--internal-val-ratio", type=float, default=0.1, help="Fraction of train-split source images held out as internal-val (task-mix only)")
    parser.add_argument("--reuse-tiles-from", type=Path, default=None, help="Existing converter output root whose tiles/ directory to reuse instead of re-extracting")
    parser.add_argument("--tile-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--recipe-name", default=None)
    parser.add_argument("--data-augment", action="store_true", help="Set data_augment=true in recipe")
    parser.add_argument("--repeat-time", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Read annotations and images but do not write tiles/jsonl")
    parser.add_argument("--image-extra-roots", nargs="*", default=[], help="Extra directories to search for images (e.g. DOTA-v1.0 images for DOTA-v2.0 labels)")
    args = parser.parse_args()
    args.image_extra_roots = [Path(p) for p in args.image_extra_roots]
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if args.tile_size % 28 != 0:
        raise ValueError("--tile-size must be a multiple of 28 for 14px patches with 2x2 merge")
    if not (0.0 <= args.overlap < 0.9):
        raise ValueError("--overlap must be in [0, 0.9)")
    if args.max_boxes_per_sample <= 0:
        raise ValueError("--max-boxes-per-sample must be positive")
    if args.min_visibility < 0:
        raise ValueError("--min-visibility must be non-negative")
    if args.min_box_size < 0:
        raise ValueError("--min-box-size must be non-negative")
    if args.samples_per_tile < 1:
        raise ValueError("--samples-per-tile must be >= 1")
    if not (0.0 < args.internal_val_ratio < 1.0):
        raise ValueError("--internal-val-ratio must be in (0, 1)")
    if args.reuse_tiles_from is not None and not args.reuse_tiles_from.is_dir():
        raise ValueError("--reuse-tiles-from must be an existing directory")
    if args.emit_rl_single_class and not args.task_mix:
        raise ValueError("--emit-rl-single-class requires the default task-mix pipeline (train 9:1 internal-val partitioning)")


def infer_version(root: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    lowered = root.name.lower()
    if "2.0" in lowered or "v2" in lowered:
        return "v2.0"
    if "1.5" in lowered or "v1.5" in lowered:
        return "v1.5"
    return "v1.0"


def choose_label_dir(split_dir: Path, version: str, label_version: str, split_name: str = "") -> Path:
    if label_version != "auto":
        candidates = [split_dir / f"labelTxt-{label_version}"]
    elif version == "v2.0":
        name = split_dir.name
        candidates = [
            split_dir / "labelTxt-v2.0" / f"DOTA-v2.0_{name}",
            split_dir / "labelTxt-v2.0",
        ]
    elif version == "v1.5":
        candidates = [split_dir / "labelTxt-v1.5", split_dir / "labelTxt-v1.0"]
    else:
        candidates = [split_dir / "labelTxt-v1.0", split_dir / "labelTxt-v1.5"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No labelTxt directory found in {split_dir}; tried {candidates}")


def index_directory_files(root: Path, suffixes: Sequence[str]) -> dict[str, Path]:
    if not root.exists():
        return {}
    indexed: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            indexed.setdefault(path.stem, path)
    return indexed


def index_zip_files(root: Path, suffixes: Sequence[str]) -> dict[str, tuple[Path, str]]:
    indexed: dict[str, tuple[Path, str]] = {}
    if not root.exists():
        return indexed
    for zip_path in root.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                suffix = Path(name).suffix.lower()
                if suffix in suffixes and not name.endswith("/"):
                    indexed.setdefault(Path(name).stem, (zip_path, name))
    return indexed


def build_records(split_dir: Path, label_dir: Path, extra_image_dirs: list[Path] | None = None) -> list[ImageRecord]:
    image_dir = split_dir / "images"
    disk_images = index_directory_files(image_dir, DOTA_EXTENSIONS)
    for extra in extra_image_dirs or []:
        for stem, path in index_directory_files(extra, DOTA_EXTENSIONS).items():
            disk_images.setdefault(stem, path)
    zip_images = index_zip_files(image_dir, DOTA_EXTENSIONS)
    disk_labels = index_directory_files(label_dir, (".txt",))
    zip_labels = index_zip_files(label_dir, (".txt",))
    image_ids = sorted(set(disk_labels) | set(zip_labels))
    records: list[ImageRecord] = []
    for image_id in image_ids:
        image_path = disk_images.get(image_id)
        image_zip = None
        image_member = None
        if image_path is None and image_id in zip_images:
            image_zip, image_member = zip_images[image_id]
        label_path = disk_labels.get(image_id)
        label_zip = None
        label_member = None
        if label_path is None and image_id in zip_labels:
            label_zip, label_member = zip_labels[image_id]
        records.append(ImageRecord(image_id, image_path, image_zip, image_member, label_path, label_zip, label_member))
    return records


def read_text_record(path: Path | None, zip_path: Path | None, member: str | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8", errors="ignore")
    if zip_path is not None and member is not None:
        with zipfile.ZipFile(zip_path) as zf:
            return zf.read(member).decode("utf-8", errors="ignore")
    raise FileNotFoundError("label file is missing")


def open_image_record(record: ImageRecord) -> Image.Image:
    if record.image_path is not None:
        return Image.open(record.image_path).convert("RGB")
    if record.image_zip is not None and record.image_member is not None:
        with zipfile.ZipFile(record.image_zip) as zf:
            with zf.open(record.image_member) as fp:
                return Image.open(fp).convert("RGB")
    raise FileNotFoundError(f"image for {record.image_id} is missing")


def parse_dota_label_text(text: str, include_difficult: bool) -> tuple[list[DotaObject], int]:
    objects: list[DotaObject] = []
    skipped_difficult = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("imagesource") or line.lower().startswith("gsd"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            coords = [float(v) for v in parts[:8]]
            difficult = int(float(parts[-1]))
        except ValueError:
            continue
        label = " ".join(parts[8:-1])
        if difficult and not include_difficult:
            skipped_difficult += 1
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        if x2 <= x1 or y2 <= y1:
            continue
        objects.append(DotaObject(label=label, difficult=difficult, hbb=(x1, y1, x2, y2)))
    return objects, skipped_difficult


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def intersect_box(box: tuple[float, float, float, float], tile: tuple[int, int, int, int]) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = box
    tx1, ty1, tx2, ty2 = tile
    ix1, iy1 = max(x1, tx1), max(y1, ty1)
    ix2, iy2 = min(x2, tx2), min(y2, ty2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def normalize_box(box: tuple[float, float, float, float], tile_x: int, tile_y: int, tile_size: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    values = (
        round((x1 - tile_x) / tile_size * 1000),
        round((y1 - tile_y) / tile_size * 1000),
        round((x2 - tile_x) / tile_size * 1000),
        round((y2 - tile_y) / tile_size * 1000),
    )
    cx1, cy1, cx2, cy2 = [max(0, min(1000, int(v))) for v in values]
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return cx1, cy1, cx2, cy2


def chunked(items: Sequence[tuple[str, tuple[int, int, int, int]]], chunk_size: int) -> Iterable[list[tuple[str, tuple[int, int, int, int]]]]:
    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def build_prompt(labels: Sequence[str]) -> str:
    return PROMPT_TEMPLATE.format(classes="</c>".join(labels))


def build_answer(boxes: Sequence[tuple[str, tuple[int, int, int, int]]]) -> str:
    if not boxes:
        return "<box>none</box>"
    parts: list[str] = []
    for label, (x1, y1, x2, y2) in boxes:
        parts.append(f"<ref>{label}</ref><box><{x1}><{y1}><{x2}><{y2}></box>")
    return "".join(parts)


def make_sample(image_rel: str, boxes: Sequence[tuple[str, tuple[int, int, int, int]]]) -> dict:
    labels = sorted({label for label, _ in boxes})
    if not labels:
        labels = ["object"]
    return {
        "conversations": [
            {"from": "human", "value": build_prompt(labels)},
            {"from": "gpt", "value": build_answer(boxes)},
        ],
        "image": image_rel,
    }


def make_negative_sample(image_rel: str, label: str) -> dict:
    """T4: query a class guaranteed absent from the tile; answer is pure none."""
    return {
        "conversations": [
            {"from": "human", "value": build_prompt([label])},
            {"from": "gpt", "value": "<box>none</box>"},
        ],
        "image": image_rel,
    }


def make_referring_sample(image_rel: str, phrase: str, box: tuple[int, int, int, int]) -> dict:
    """T5: single-instance referring expression; prompt matches LocateAnythingWorker.ground_single."""
    x1, y1, x2, y2 = box
    return {
        "conversations": [
            {"from": "human", "value": SINGLE_PROMPT_TEMPLATE.format(phrase=phrase)},
            {"from": "gpt", "value": f"<ref>{phrase}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"},
        ],
        "image": image_rel,
    }


def make_rl_single_class_samples(
    image_rel: str, kept: Sequence[tuple[str, tuple[int, int, int, int]]]
) -> list[dict]:
    """RL canonical GT: one row per (tile, single-class prompt) with the FULL box set.

    Unlike emit_tile_samples()/make_sample() this never chunks and never mixes
    tasks: each answer contains every kept box of that label in the tile, so
    dense-tile GT survives intact for the RL reward.
    """
    by_label: dict[str, list[tuple[int, int, int, int]]] = {}
    for label, box in kept:
        by_label.setdefault(label, []).append(box)
    samples: list[dict] = []
    for label in sorted(by_label):
        boxes = by_label[label]
        samples.append(
            {
                "conversations": [
                    {"from": "human", "value": build_prompt([label])},
                    {"from": "gpt", "value": build_answer([(label, b) for b in boxes])},
                ],
                "image": image_rel,
            }
        )
    return samples


def intersecting_labels(objects: Sequence[DotaObject], tile_box: tuple[int, int, int, int]) -> set[str]:
    """Labels of every object whose HBB intersects the tile, before visibility/min-box filters."""
    return {obj.label for obj in objects if intersect_box(obj.hbb, tile_box) is not None}


def guaranteed_absent(split_classes: set[str], intersecting: set[str]) -> set[str]:
    """Classes provably absent from this tile: split inventory minus anything intersecting it."""
    return set(split_classes) - set(intersecting)


def pick_t5(kept: Sequence[tuple[str, tuple[int, int, int, int]]]) -> tuple[str, tuple[int, int, int, int]] | None:
    """Pick a referring target: uniform label, uniform predicate, extremal box (ties: first in kept order)."""
    if not kept:
        return None
    label = random.choice(sorted({item_label for item_label, _ in kept}))
    predicate = random.choice(T5_PREDICATES)
    boxes = [box for item_label, box in kept if item_label == label]
    if predicate == "leftmost":
        box = min(boxes, key=lambda b: b[0] + b[2])
    elif predicate == "rightmost":
        box = max(boxes, key=lambda b: b[0] + b[2])
    elif predicate == "topmost":
        box = min(boxes, key=lambda b: b[1] + b[3])
    else:
        box = max(boxes, key=lambda b: b[1] + b[3])
    return f"the {predicate} {label}", box


def emit_tile_samples(
    args: argparse.Namespace,
    write_line,
    tile_rel: str,
    kept: list[tuple[str, tuple[int, int, int, int]]],
    split_classes: set[str],
    intersecting: set[str],
) -> tuple[int, int, Counter, Counter]:
    """Emit `samples_per_tile` mixed-task draws for one non-empty tile.

    A draw that cannot be resolved (e.g. T4 with no absent class) is re-drawn up
    to 10 times and then skipped. Returns (n_samples, n_kept_objects,
    class_counts, task_counts).
    """
    present = {label for label, _ in kept}
    absent = guaranteed_absent(split_classes, intersecting)
    n_samples = 0
    n_kept = 0
    class_counts: Counter = Counter()
    task_counts: Counter = Counter()
    for _draw in range(args.samples_per_tile):
        for _attempt in range(10):
            task = random.choices(list(TASK_MIX_WEIGHTS), weights=list(TASK_MIX_WEIGHTS.values()), k=1)[0]
            if task == "t4":
                eligible = [c for c in sorted(absent) if c not in T4_SCENE_SCALE_SKIP]
                if not eligible:
                    eligible = sorted(absent)
                if not eligible:
                    continue
                if write_line is not None:
                    write_line(make_negative_sample(tile_rel, random.choice(eligible)))
                n_samples += 1
                task_counts["t4"] += 1
                break
            if task == "t5":
                picked = pick_t5(kept)
                if picked is None:
                    continue
                phrase, box = picked
                if write_line is not None:
                    write_line(make_referring_sample(tile_rel, phrase, box))
                n_samples += 1
                task_counts["t5"] += 1
                break
            if task == "t1":
                if not kept:
                    continue
                boxes = list(kept)
            elif task == "t2":
                if not present:
                    continue
                chosen_label = random.choice(sorted(present))
                boxes = [(l, b) for l, b in kept if l == chosen_label]
            else:  # t3
                if not present:
                    continue
                if len(present) < 2:
                    chosen_label = random.choice(sorted(present))
                    boxes = [(l, b) for l, b in kept if l == chosen_label]
                else:
                    chosen = set(random.sample(sorted(present), random.randint(2, len(present))))
                    boxes = [(l, b) for l, b in kept if l in chosen]
            for box_chunk in chunked(boxes, args.max_boxes_per_sample):
                if write_line is not None:
                    write_line(make_sample(tile_rel, box_chunk))
                n_samples += 1
                n_kept += len(box_chunk)
                class_counts.update(label for label, _ in box_chunk)
                task_counts[task] += 1
            break
    return n_samples, n_kept, class_counts, task_counts


def save_tile(image: Image.Image, out_path: Path, tile_box: tuple[int, int, int, int], tile_format: str, jpeg_quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile = image.crop(tile_box)
    if tile_format == "jpg":
        tile.save(out_path, format="JPEG", quality=jpeg_quality)
    else:
        tile.save(out_path, format="PNG")


def _relative_or_abs(path: Path, cwd: Path) -> str:
    try:
        return str(path.resolve().relative_to(cwd))
    except ValueError:
        return str(path.resolve())

def convert_split(args: argparse.Namespace, version: str, split: str, recipe_entries: dict, partition_of: dict, cwd: Path, rl_single_outputs: dict) -> SplitStats:
    split_dir = args.dota_root / split
    label_dir = choose_label_dir(split_dir, version, args.label_version)
    records = build_records(split_dir, label_dir, args.image_extra_roots)
    stats = SplitStats(split=split)

    # Label-only first pass over the whole split: full class inventory so T4
    # negatives draw from the split's real class set (v1.0/v1.5/v2.0 follow the data).
    split_classes: set[str] = set()
    for record in records:
        if record.label_path is None and record.label_zip is None:
            continue
        label_text = read_text_record(record.label_path, record.label_zip, record.label_member)
        first_pass_objects, _ = parse_dota_label_text(label_text, args.include_difficult)
        split_classes.update(obj.label for obj in first_pass_objects)

    # 9:1 source-image holdout on the DOTA train split (task-mix only).
    internal_val_ids: set[str] = set()
    if split == "train" and args.task_mix and len(records) >= 2:
        random.shuffle(records)
        n_val = int(round(len(records) * args.internal_val_ratio))
        n_val = max(1, min(n_val, len(records) - 1))
        internal_val_ids = {record.image_id for record in records[-n_val:]}
    if split == "train":
        stats.n_train_images = len(records) - len(internal_val_ids)
        stats.n_internal_val_images = len(internal_val_ids)

    # Output partitions: train split -> train_mix + internal_val_mix; other
    # splits -> dual test renders (all-classes T1 scorer file + mixed tasks).
    stem = args.dota_root.name
    if args.task_mix:
        if split == "train":
            partition_specs: list[tuple[str, str]] = [
                ("train_mix", f"{stem}_train_mix_hbb_{args.tile_size}.jsonl"),
                ("internal_val_mix", f"{stem}_internal_val_mix_hbb_{args.tile_size}.jsonl"),
            ]
            if args.emit_rl_single_class:
                partition_specs += [
                    ("train_rl_single", f"{stem}_train_rl_single_hbb_{args.tile_size}.jsonl"),
                    ("internal_val_rl_single", f"{stem}_internal_val_rl_single_hbb_{args.tile_size}.jsonl"),
                ]
        else:
            partition_specs = [
                ("test_t1", f"{stem}_test_t1_hbb_{args.tile_size}.jsonl"),
                ("test_mix", f"{stem}_test_mix_hbb_{args.tile_size}.jsonl"),
            ]
    else:
        partition_specs = [(split, f"{stem}_{split}_hbb_{args.tile_size}.jsonl")]

    def partition_has_images(partition: str) -> bool:
        if not args.task_mix or split != "train":
            return bool(records)
        if partition == "internal_val_mix":
            return bool(internal_val_ids)
        if partition == "internal_val_rl_single":
            return bool(internal_val_ids)
        if partition == "train_rl_single":
            return len(records) > len(internal_val_ids)
        return len(records) > len(internal_val_ids)

    jsonl_fhs: dict[str, object] = {}
    partition_files: dict[str, Path] = {}
    if not args.dry_run:
        for partition, filename in partition_specs:
            if not partition_has_images(partition):
                continue
            annotation_path = args.output_root / "annotations" / filename
            annotation_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_fhs[partition] = annotation_path.open("w", encoding="utf-8")
            partition_files[partition] = annotation_path

    tiles_base = args.reuse_tiles_from if args.reuse_tiles_from is not None else args.output_root
    if not args.dry_run and args.reuse_tiles_from is None:
        (args.output_root / "tiles" / split).mkdir(parents=True, exist_ok=True)
    try:
        for record in records:
            stats.images_seen += 1
            if record.image_path is None and record.image_zip is None:
                stats.images_missing += 1
                continue
            label_text = read_text_record(record.label_path, record.label_zip, record.label_member)
            objects, skipped_difficult = parse_dota_label_text(label_text, args.include_difficult)
            stats.skipped_difficult_objects += skipped_difficult
            stats.source_objects += len(objects)
            if not objects and not args.include_empty:
                stats.images_without_objects += 1
                continue
            try:
                image = open_image_record(record)
            except FileNotFoundError:
                stats.images_missing += 1
                continue
            width, height = image.size
            x_starts = tile_starts(width, args.tile_size, args.overlap)
            y_starts = tile_starts(height, args.tile_size, args.overlap)
            for tile_y in y_starts:
                for tile_x in x_starts:
                    tile_box = (tile_x, tile_y, tile_x + args.tile_size, tile_y + args.tile_size)
                    kept: list[tuple[str, tuple[int, int, int, int]]] = []
                    for obj in objects:
                        clipped = intersect_box(obj.hbb, tile_box)
                        if clipped is None:
                            continue
                        original_area = (obj.hbb[2] - obj.hbb[0]) * (obj.hbb[3] - obj.hbb[1])
                        clipped_area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
                        if original_area <= 0 or clipped_area / original_area < args.min_visibility:
                            stats.skipped_small_or_invisible_objects += 1
                            continue
                        if (clipped[2] - clipped[0]) < args.min_box_size or (clipped[3] - clipped[1]) < args.min_box_size:
                            stats.skipped_small_or_invisible_objects += 1
                            continue
                        norm_box = normalize_box(clipped, tile_x, tile_y, args.tile_size)
                        if norm_box is None:
                            stats.skipped_small_or_invisible_objects += 1
                            continue
                        kept.append((obj.label, norm_box))
                    if not kept and not args.include_empty:
                        continue
                    suffix = "jpg" if args.tile_format == "jpg" else "png"
                    tile_name = f"{record.image_id}__x{tile_x}_y{tile_y}_s{args.tile_size}.{suffix}"
                    tile_rel = str(Path("tiles") / split / tile_name)
                    tile_path = tiles_base / tile_rel
                    if args.reuse_tiles_from is not None:
                        if not tile_path.exists():
                            stats.tiles_missing += 1
                            continue
                        stats.tiles_written += 1
                    elif not args.dry_run and not tile_path.exists():
                        save_tile(image, tile_path, tile_box, args.tile_format, args.jpeg_quality)
                        stats.tiles_written += 1
                    elif tile_path.exists():
                        stats.tiles_written += 1
                    intersecting = intersecting_labels(objects, tile_box)
                    # Empty tiles are not used as T4 factories: mix emission needs kept boxes.
                    emissions: list[tuple[str, object]] = []
                    if args.task_mix:
                        if split == "train":
                            if kept:
                                partition = "internal_val_mix" if record.image_id in internal_val_ids else "train_mix"
                                emissions.append(("mix", jsonl_fhs.get(partition)))
                        else:
                            emissions.append(("legacy", jsonl_fhs.get("test_t1")))
                            if kept:
                                emissions.append(("mix", jsonl_fhs.get("test_mix")))
                    else:
                        emissions.append(("legacy", jsonl_fhs.get(split)))
                    for mode, fh in emissions:
                        if mode == "legacy":
                            for box_chunk in chunked(kept, args.max_boxes_per_sample):
                                if not args.dry_run and fh is not None:
                                    fh.write(json.dumps(make_sample(tile_rel, box_chunk), ensure_ascii=False) + "\n")
                                stats.samples_written += 1
                                stats.kept_objects += len(box_chunk)
                                stats.class_counts.update(label for label, _ in box_chunk)
                                stats.task_counts["t1"] += 1
                        else:
                            write_line = None
                            if not args.dry_run and fh is not None:
                                write_line = lambda sample, fh=fh: fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                            n_samples, n_kept, cls_counts, task_counts = emit_tile_samples(
                                args, write_line, tile_rel, kept, split_classes, intersecting
                            )
                            stats.samples_written += n_samples
                            stats.kept_objects += n_kept
                            stats.class_counts.update(cls_counts)
                            stats.task_counts.update(task_counts)
                    if args.emit_rl_single_class and split == "train" and kept:
                        rl_partition = (
                            "internal_val_rl_single"
                            if record.image_id in internal_val_ids
                            else "train_rl_single"
                        )
                        rl_fh = jsonl_fhs.get(rl_partition)
                        rl_rows = make_rl_single_class_samples(tile_rel, kept)
                        if not args.dry_run and rl_fh is not None:
                            for rl_row in rl_rows:
                                rl_fh.write(json.dumps(rl_row, ensure_ascii=False) + "\n")
                        stats.rl_single_samples_written += len(rl_rows)
                        for rl_row in rl_rows:
                            n_boxes = rl_row["conversations"][1]["value"].count("<ref>")
                            stats.rl_single_max_boxes = max(stats.rl_single_max_boxes, n_boxes)
                        stats.rl_single_class_counts.update(label for label, _ in kept)
            image.close()
    finally:
        for fh in jsonl_fhs.values():
            fh.close()

    if not args.dry_run:
        base_key = args.recipe_name or args.dota_root.name
        root_recipe = _relative_or_abs(args.output_root, cwd)
        for partition, _filename in partition_specs:
            annotation_path = partition_files.get(partition)
            if annotation_path is None:
                continue
            dataset_key = f"{base_key}_{partition}_hbb_{args.tile_size}"
            if partition.endswith("_rl_single"):
                rl_single_outputs["annotations"][partition] = _relative_or_abs(annotation_path, cwd)
                if partition == "train_rl_single":
                    rl_single_outputs["recipe"][dataset_key] = {
                        "annotation": _relative_or_abs(annotation_path, cwd),
                        "root": root_recipe,
                        "repeat_time": args.repeat_time,
                        "data_augment": args.data_augment,
                    }
                continue
            recipe_entries[dataset_key] = {
                "annotation": _relative_or_abs(annotation_path, cwd),
                "root": root_recipe,
                "repeat_time": args.repeat_time,
                "data_augment": args.data_augment,
            }
            partition_of[dataset_key] = partition
    return stats


def write_recipe_and_metadata(args: argparse.Namespace, version: str, recipe_entries: dict, partition_of: dict, stats: list[SplitStats], rl_single_outputs: dict) -> None:
    if args.dry_run:
        return
    recipes_dir = args.output_root / "recipes"
    metadata_dir = args.output_root / "metadata"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    recipe_name = args.recipe_name or f"{args.dota_root.name}_hbb_{args.tile_size}"

    # Tiles-reuse mode: link output_root/tiles at the reused tiles/ so recipe
    # `root` resolves `tiles/...` image paths without copying pixel data.
    if args.reuse_tiles_from is not None:
        tiles_link = args.output_root / "tiles"
        if not tiles_link.exists() and not tiles_link.is_symlink():
            tiles_link.symlink_to((args.reuse_tiles_from / "tiles").resolve())

    def write_recipe(filename: str, entries: dict) -> None:
        if entries:
            (recipes_dir / filename).write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def subset(partitions: tuple[str, ...]) -> dict:
        return {key: value for key, value in recipe_entries.items() if partition_of.get(key) in partitions}

    if args.task_mix:
        write_recipe(f"{recipe_name}.json", subset(("train_mix", "internal_val_mix")))
        write_recipe(f"{recipe_name}_train_only.json", subset(("train_mix",)))
        write_recipe(f"{recipe_name}_internal_val.json", subset(("internal_val_mix",)))
        write_recipe(f"{recipe_name}_test_t1.json", subset(("test_t1",)))
        write_recipe(f"{recipe_name}_test_mix.json", subset(("test_mix",)))
    else:
        write_recipe(f"{recipe_name}.json", dict(recipe_entries))
        write_recipe(f"{recipe_name}_train_only.json", {key: value for key, value in recipe_entries.items() if "_train_" in key})
    if args.emit_rl_single_class:
        write_recipe(f"{recipe_name}_rl_single.json", dict(rl_single_outputs["recipe"]))

    metadata = {
        "dota_root": str(args.dota_root),
        "version": version,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "min_visibility": args.min_visibility,
        "min_box_size": args.min_box_size,
        "max_boxes_per_sample": args.max_boxes_per_sample,
        "include_difficult": args.include_difficult,
        "tile_format": args.tile_format,
        "task_mix": args.task_mix,
        "samples_per_tile": args.samples_per_tile,
        "internal_val_ratio": args.internal_val_ratio,
        "task_mix_weights": dict(TASK_MIX_WEIGHTS),
        "reuse_tiles_from": str(args.reuse_tiles_from) if args.reuse_tiles_from is not None else None,
        "t4_scene_scale_skip": sorted(T4_SCENE_SCALE_SKIP),
        "emit_rl_single_class": args.emit_rl_single_class,
        "rl_single_annotations": dict(rl_single_outputs["annotations"]),
        "splits": [s.to_dict() for s in stats],
    }
    metadata_path = metadata_dir / f"{recipe_name}_stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    version = infer_version(args.dota_root, args.version)
    cwd = Path.cwd().resolve()
    recipe_entries: dict = {}
    partition_of: dict = {}
    rl_single_outputs: dict = {"recipe": {}, "annotations": {}}
    all_stats: list[SplitStats] = []
    for split in args.splits:
        stats = convert_split(args, version, split, recipe_entries, partition_of, cwd, rl_single_outputs)
        all_stats.append(stats)
        print(json.dumps(stats.to_dict(), ensure_ascii=False), flush=True)
    write_recipe_and_metadata(args, version, recipe_entries, partition_of, all_stats, rl_single_outputs)


if __name__ == "__main__":
    main()
