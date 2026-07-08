#!/usr/bin/env python
"""Convert a pretrained LocateAnything checkpoint from 2x2 MoonViT merge to 1x1.

The MLP projector input dimension changes from vit_hidden_size*4 to
vit_hidden_size. This script surgically averages the old 2x2 MLP weights into
the new 1x1 MLP shape, giving a much better initialization than random weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
from eaglevl.model.locany.modeling_locateanything import (
    LocateAnythingForConditionalGeneration,
    build_mlp_projector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True, help="Original LocateAnything checkpoint path")
    parser.add_argument("--output-path", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument("--merge-kernel", type=str, default="1,1", help="Target merge kernel size, e.g. 1,1")
    return parser.parse_args()


def average_mlp_weights(old_mlp, new_mlp, old_merge_product: int, new_merge_product: int) -> None:
    assert old_merge_product % new_merge_product == 0, "old_merge_product must be divisible by new_merge_product"
    ratio = old_merge_product // new_merge_product

    # LayerNorm: average groups along the hidden dimension
    old_ln = old_mlp[0]
    new_ln = new_mlp[0]
    with torch.no_grad():
        new_ln.weight.copy_(old_ln.weight.view(-1, ratio).mean(dim=1))
        new_ln.bias.copy_(old_ln.bias.view(-1, ratio).mean(dim=1))

    # First Linear: average input-channel groups
    old_linear1 = old_mlp[1]
    new_linear1 = new_mlp[1]
    out_features, old_in_features = old_linear1.weight.shape
    new_in_features = old_in_features // ratio
    with torch.no_grad():
        new_linear1.weight.copy_(
            old_linear1.weight.view(out_features, new_in_features, ratio).mean(dim=2)
        )
        new_linear1.bias.copy_(old_linear1.bias)

    # GELU has no weights
    # Second Linear: dimensions unchanged, copy directly
    old_linear2 = old_mlp[3]
    new_linear2 = new_mlp[3]
    with torch.no_grad():
        new_linear2.weight.copy_(old_linear2.weight)
        new_linear2.bias.copy_(old_linear2.bias)


def main() -> None:
    args = parse_args()
    merge_kernel = tuple(int(x) for x in args.merge_kernel.split(","))
    if len(merge_kernel) != 2 or any(k <= 0 for k in merge_kernel):
        raise ValueError("--merge-kernel must be two positive integers like '1,1'")
    merge_product = merge_kernel[0] * merge_kernel[1]

    config = LocateAnythingConfig.from_pretrained(args.input_path)
    config.vision_config.merge_kernel_size = list(merge_kernel)

    # Force SDPA so the script can run on CPU / non-magi machines.
    for cfg in (config, config.vision_config, config.text_config):
        cfg._attn_implementation = "sdpa"
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = "sdpa"

    print(f"Loading original model from {args.input_path} ...")
    model = LocateAnythingForConditionalGeneration.from_pretrained(
        args.input_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        ignore_mismatched_sizes=True,
    )
    old_merge_kernel = getattr(model, "visual_merge_kernel_size", (2, 2))
    old_merge_product = old_merge_kernel[0] * old_merge_kernel[1]
    if old_merge_product != 4:
        print(f"Warning: expected original merge kernel product 4, got {old_merge_product}")

    print("Rebuilding MLP projector for merge kernel", merge_kernel)
    old_mlp = model.mlp1
    model.mlp1 = build_mlp_projector(config.vision_config, config.text_config)
    average_mlp_weights(old_mlp, model.mlp1, old_merge_product, merge_product)
    model.visual_merge_kernel_size = merge_kernel

    print(f"Saving converted model to {args.output_path} ...")
    args.output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_path)

    # Copy tokenizer / processor / custom-code files so the converted dir is loadable.
    source_files = [
        "configuration_locateanything.py", "configuration_qwen2.py",
        "modeling_locateanything.py", "modeling_qwen2.py", "modeling_vit.py",
        "generate_utils.py", "image_processing_locateanything.py",
        "processing_locateanything.py", "mask_magi_utils.py", "mask_sdpa_utils.py",
        "added_tokens.json", "chat_template.json", "generation_config.json",
        "merges.txt", "preprocessor_config.json", "processor_config.json",
        "special_tokens_map.json", "tokenizer_config.json", "vocab.json",
        "README.md", "batch_infer.py",
    ]
    for name in source_files:
        src = args.input_path / name
        dst = args.output_path / name
        if src.exists() and not dst.exists():
            import shutil
            shutil.copy2(src, dst)

    print("Done.")


if __name__ == "__main__":
    main()
