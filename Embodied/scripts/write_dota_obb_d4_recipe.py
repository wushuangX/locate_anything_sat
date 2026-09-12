#!/usr/bin/env python
"""Expand a converter 0° OBB train_only recipe into the D4 eight-key recipe.

Identity key keeps T5. The seven geom_op keys point at
`{stem}_{op}.jsonl` written by `augment_locany_jsonl.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from eaglevl.train.augmentation import GEOM_OPS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zero-recipe", type=Path, required=True, help="Converter *_train_only.json (one identity key)")
    p.add_argument("--output", type=Path, required=True, help="Destination recipe JSON (may overwrite --zero-recipe)")
    return p.parse_args()


def expand_identity_recipe(zero: dict) -> dict:
    if len(zero) != 1:
        raise ValueError(f"expected exactly one identity key, got {list(zero)}")
    ident_key, ident = next(iter(zero.items()))
    base = dict(ident)
    base["geometry"] = "obb"
    base["color_jitter"] = True
    base.pop("geom_op", None)
    base.pop("rotate", None)
    base.pop("hflip", None)
    base.pop("flip_before_rotate", None)

    ann0 = Path(base["annotation"])
    out = {ident_key: dict(base)}
    for op in GEOM_OPS:
        entry = dict(base)
        entry["annotation"] = str(ann0.with_name(f"{ann0.stem}_{op}{ann0.suffix}"))
        entry["geom_op"] = op
        out[f"{ident_key}_{op}"] = entry
    return out


def main() -> None:
    args = parse_args()
    zero = json.loads(args.zero_recipe.read_text(encoding="utf-8"))
    expanded = expand_identity_recipe(zero)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(expanded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[recipe] wrote {len(expanded)} keys -> {args.output}")

if __name__ == "__main__":
    main()
