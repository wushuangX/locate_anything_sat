#!/usr/bin/env python
"""CPU smoke: mixed HBB/OBB MTP packing masks and handle_pattern."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

_repo_root = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path, package: str | None = None):
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=[str(path.parent)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_mask_modules():
    locany_dir = _repo_root / "eaglevl" / "model" / "locany"
    pkg = types.ModuleType("locany_masks")
    pkg.__path__ = [str(locany_dir)]
    sys.modules["locany_masks"] = pkg
    sdpa = _load("locany_masks.mask_sdpa_utils", locany_dir / "mask_sdpa_utils.py")
    magi = _load("locany_masks.mask_magi_utils", locany_dir / "mask_magi_utils.py")
    return magi, sdpa


def _load_generate_utils():
    return _load(
        "generate_utils_smoke",
        _repo_root / "eaglevl" / "utils" / "locany" / "generate_utils.py",
    )


def _packed_ids() -> tuple[torch.Tensor, torch.Tensor]:
    pos0 = list(range(8)) + list(range(3, 9))
    pos1 = list(range(8)) + list(range(3, 10))
    position_ids = torch.tensor([pos0 + pos1], dtype=torch.long)
    data_index = torch.tensor([[0] * 14 + [1] * 15], dtype=torch.long)
    return position_ids, data_index


def check_magi_plan(convert_mtp_mask_to_magi_plan) -> None:
    position_ids, data_index = _packed_ids()
    plan = convert_mtp_mask_to_magi_plan(
        block_size=6,
        position_ids=position_ids,
        data_index=data_index,
        sample_block_sizes=torch.tensor([6, 7]),
    )
    q = plan["q_ranges"].tolist()
    mtp_lens = {e - s for s, e in q if (s, e) in ((8, 14), (22, 29))}
    assert mtp_lens == {6, 7}, q

    plan_bad = convert_mtp_mask_to_magi_plan(
        block_size=6,
        position_ids=position_ids,
        data_index=data_index,
        sample_block_sizes=None,
    )
    s1_mtp = [(s, e) for s, e in plan_bad["q_ranges"].tolist() if s >= 22]
    assert any((e - s) == 6 for s, e in s1_mtp), s1_mtp
    assert any((e - s) == 1 for s, e in s1_mtp), s1_mtp


def check_sdpa_mask(create_mtp_packing_mask_4d) -> None:
    position_ids, data_index = _packed_ids()
    mask = create_mtp_packing_mask_4d(
        block_size=6,
        position_ids=position_ids,
        data_index=data_index,
        sample_block_sizes=torch.tensor([6, 7]),
    )
    m = mask[0, 0]
    assert (m[8:14, 8:14] == 0).all()
    assert (m[22:29, 22:29] == 0).all()
    assert torch.isneginf(m[8, 22])


def check_handle_pattern(handle_pattern) -> None:
    token_ids = {
        "null_token_id": 0,
        "im_end_token_id": 1,
        "box_start_token_id": 10,
        "box_end_token_id": 11,
        "none_token_id": 12,
        "coord_start_token_id": 100,
        "coord_end_token_id": 1100,
        "ref_end_token_id": 13,
        "obb_start_token_id": 20,
        "obb_end_token_id": 21,
    }
    hbb = [10, 200, 201, 202, 203, 11]
    out = handle_pattern(hbb, token_ids, "hybrid")
    assert out["type"] == "coord_box", out
    obb = [20, 200, 201, 202, 203, 204, 21]
    out = handle_pattern(obb, token_ids, "hybrid")
    assert out["type"] == "coord_obb", out
    assert out["tokens"] == obb


def main() -> None:
    magi, sdpa = _load_mask_modules()
    gen = _load_generate_utils()
    check_magi_plan(magi.convert_mtp_mask_to_magi_plan)
    check_sdpa_mask(sdpa.create_mtp_packing_mask_4d)
    check_handle_pattern(gen.handle_pattern)
    print("pbd mixed block size smoke passed")


if __name__ == "__main__":
    main()
