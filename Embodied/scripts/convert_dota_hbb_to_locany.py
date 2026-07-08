#!/usr/bin/env python
"""Convert DOTA OBB annotations to tiled LocateAnything HBB JSONL.

The converter supports extracted DOTA-v1.0, DOTA-v1.5 and DOTA-v2.0 layouts.
It also supports v1.x image/label directories whose files are still inside zip
archives by reading from zip and writing 512x512 training tiles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

PROMPT_TEMPLATE = "Locate all the instances that matches the following description: {classes}."
DOTA_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


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

    def __post_init__(self) -> None:
        if self.class_counts is None:
            self.class_counts = Counter()

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
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dota-root", type=Path, required=True, help="DOTA dataset root, e.g. /data/.../DOTA-v1.0")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for tiles, JSONL, recipe and metadata")
    parser.add_argument("--version", choices=["auto", "v1.0", "v1.5", "v2.0"], default="auto")
    parser.add_argument("--splits", nargs="+", default=["train"], help="Labelled splits to convert, e.g. train val")
    parser.add_argument("--label-version", choices=["auto", "v1.0", "v1.5", "v2.0"], default="auto")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=float, default=0.0, help="Tile overlap ratio in [0, 0.9)")
    parser.add_argument("--min-visibility", type=float, default=0.5, help="Min clipped HBB area/original HBB area")
    parser.add_argument("--min-box-size", type=float, default=2.0, help="Min clipped width/height in pixels")
    parser.add_argument("--max-boxes-per-sample", type=int, default=30)
    parser.add_argument("--include-difficult", action="store_true", default=True)
    parser.add_argument("--exclude-difficult", action="store_false", dest="include_difficult")
    parser.add_argument("--include-empty", action="store_true", help="Write <box>none</box> samples for empty tiles")
    parser.add_argument("--tile-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--recipe-name", default=None)
    parser.add_argument("--data-augment", action="store_true", help="Set data_augment=true in recipe")
    parser.add_argument("--repeat-time", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Read annotations and images but do not write tiles/jsonl")
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")
    if not (0.0 <= args.overlap < 0.9):
        raise ValueError("--overlap must be in [0, 0.9)")
    if args.max_boxes_per_sample <= 0:
        raise ValueError("--max-boxes-per-sample must be positive")
    if args.min_visibility < 0:
        raise ValueError("--min-visibility must be non-negative")
    if args.min_box_size < 0:
        raise ValueError("--min-box-size must be non-negative")


def infer_version(root: Path, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    lowered = root.name.lower()
    if "2.0" in lowered or "v2" in lowered:
        return "v2.0"
    if "1.5" in lowered or "v1.5" in lowered:
        return "v1.5"
    return "v1.0"


def choose_label_dir(split_dir: Path, version: str, label_version: str) -> Path:
    if label_version != "auto":
        candidates = [split_dir / f"labelTxt-{label_version}"]
    elif version == "v2.0":
        candidates = [split_dir / "labelTxt-v2.0"]
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


def build_records(split_dir: Path, label_dir: Path) -> list[ImageRecord]:
    image_dir = split_dir / "images"
    disk_images = index_directory_files(image_dir, DOTA_EXTENSIONS)
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


def save_tile(image: Image.Image, out_path: Path, tile_box: tuple[int, int, int, int], tile_format: str, jpeg_quality: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile = image.crop(tile_box)
    if tile_format == "jpg":
        tile.save(out_path, format="JPEG", quality=jpeg_quality)
    else:
        tile.save(out_path, format="PNG")


def convert_split(args: argparse.Namespace, version: str, split: str, recipe_entries: dict, cwd: Path) -> SplitStats:
    split_dir = args.dota_root / split
    label_dir = choose_label_dir(split_dir, version, args.label_version)
    records = build_records(split_dir, label_dir)
    stats = SplitStats(split=split)

    annotation_rel = Path("annotations") / f"{args.dota_root.name}_{split}_hbb_{args.tile_size}.jsonl"
    annotation_path = args.output_root / annotation_rel
    tile_dir = args.output_root / "tiles" / split
    if not args.dry_run:
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        tile_dir.mkdir(parents=True, exist_ok=True)

    jsonl_fh = None if args.dry_run else annotation_path.open("w", encoding="utf-8")
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
                    tile_path = args.output_root / tile_rel
                    if not args.dry_run and not tile_path.exists():
                        save_tile(image, tile_path, tile_box, args.tile_format, args.jpeg_quality)
                        stats.tiles_written += 1
                    elif tile_path.exists():
                        stats.tiles_written += 1
                    for box_chunk in chunked(kept, args.max_boxes_per_sample):
                        if not args.dry_run and jsonl_fh is not None:
                            jsonl_fh.write(json.dumps(make_sample(tile_rel, box_chunk), ensure_ascii=False) + "\n")
                        stats.samples_written += 1
                        stats.kept_objects += len(box_chunk)
                        stats.class_counts.update(label for label, _ in box_chunk)
            image.close()
    finally:
        if jsonl_fh is not None:
            jsonl_fh.close()

    if not args.dry_run:
        dataset_key = f"{args.recipe_name or args.dota_root.name}_{split}_hbb_{args.tile_size}"
        annotation_recipe = str(annotation_path.resolve().relative_to(cwd))
        root_recipe = str(args.output_root.resolve().relative_to(cwd))
        recipe_entries[dataset_key] = {
            "annotation": annotation_recipe,
            "root": root_recipe,
            "repeat_time": args.repeat_time,
            "data_augment": args.data_augment,
        }
    return stats


def write_recipe_and_metadata(args: argparse.Namespace, version: str, recipe_entries: dict, stats: list[SplitStats]) -> None:
    if args.dry_run:
        return
    recipes_dir = args.output_root / "recipes"
    metadata_dir = args.output_root / "metadata"
    recipes_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    recipe_name = args.recipe_name or f"{args.dota_root.name}_hbb_{args.tile_size}"
    recipe_path = recipes_dir / f"{recipe_name}.json"
    recipe_path.write_text(json.dumps(recipe_entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
        "splits": [s.to_dict() for s in stats],
    }
    metadata_path = metadata_dir / f"{recipe_name}_stats.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    version = infer_version(args.dota_root, args.version)
    cwd = Path.cwd().resolve()
    recipe_entries: dict = {}
    all_stats: list[SplitStats] = []
    for split in args.splits:
        stats = convert_split(args, version, split, recipe_entries, cwd)
        all_stats.append(stats)
        print(json.dumps(stats.to_dict(), ensure_ascii=False), flush=True)
    write_recipe_and_metadata(args, version, recipe_entries, all_stats)


if __name__ == "__main__":
    main()
