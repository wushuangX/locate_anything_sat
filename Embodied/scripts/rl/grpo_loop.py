#!/usr/bin/env python
"""RL Phase B on-policy GRPO loop（HBB 100k，NTP，无 KL / 无 DeepSpeed）。

采样分布锁在 rollout.RL_GENERATION_KW（slow / top_p=1 / top_k=0 / rep=1）。
组内优势 Â_i = (r_i − mean) / (std + eps)；std≈0 则 skip 本 step。
只训练已有 LLM LoRA；vision / mlp 冻结。不改 locany_finetune_magi_stream.py。
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import sys
from pathlib import Path
from random import Random

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Embodied 根
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/rl

import freeze_check  # noqa: E402
import reward  # noqa: E402
import rollout  # noqa: E402

TILE_SIZE = freeze_check.TILE_SIZE


def grpo_advantages(rewards: list[float], eps: float = 1e-8) -> tuple[list[float], bool]:
    """组内标准化优势。std < eps → 全 0 且 skip=True（本 step 不 optimizer.step）。"""
    if len(rewards) == 0:
        return [], True
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std < eps:
        return [0.0] * len(rewards), True
    return [(r - mean) / (std + eps) for r in rewards], False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        default="work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1",
    )
    p.add_argument(
        "--ann",
        default="data/dota_v1_hbb_448_mix_v1/annotations/DOTA-v1.0_train_mix_hbb_448.jsonl",
    )
    p.add_argument("--data-root", default="data/dota_v1_hbb_448_mix_v1")
    p.add_argument("--output-dir", default="work_dirs/rl_grpo_hbb_100k")
    p.add_argument("--n", type=int, default=8, help="组大小 G")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def _collect_lora_params(model):
    return [(n, p) for n, p in model.named_parameters() if "lora_" in n]


def _prepare_policy(model):
    named = _collect_lora_params(model)
    if not named:
        print(f"[grpo] language_model type={type(model.language_model)!r}", flush=True)
        try:
            from peft import PeftModel
            is_peft = isinstance(model.language_model, PeftModel)
        except ImportError:
            is_peft = "Peft" in type(model.language_model).__name__
        if not is_peft:
            model.wrap_llm_lora(r=64, lora_alpha=128)
            named = _collect_lora_params(model)
        if not named:
            raise RuntimeError("no lora_ parameters found after load")

    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n
    model.vision_model.requires_grad_(False)
    model.mlp1.requires_grad_(False)
    model.vision_model.eval()
    model.mlp1.eval()
    lora_params = [p for _, p in named if p.requires_grad]
    if not lora_params:
        raise RuntimeError("no trainable lora_ parameters")
    n_train = sum(p.numel() for p in lora_params)
    print(f"[grpo] trainable lora params={n_train}", flush=True)
    return lora_params


def completion_logprob(model, processor, image, prompt, sequences, device, dtype, worker):
    messages = worker._build_messages(image, prompt)
    text = processor.py_apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device=device, dtype=dtype)
    image_grid_hws = inputs.get("image_grid_hws", None)
    if image_grid_hws is not None:
        if not torch.is_tensor(image_grid_hws):
            image_grid_hws = torch.as_tensor(image_grid_hws, dtype=torch.int32)
        image_grid_hws = image_grid_hws.to(device=device, dtype=torch.int32)

    comp = sequences.view(-1).to(device)
    if comp.numel() == 0:
        return torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    full_ids = torch.cat([prompt_ids, comp.unsqueeze(0)], dim=1)
    attn = torch.ones(full_ids.shape, dtype=torch.long, device=device)
    out = model(
        pixel_values=pixel_values,
        input_ids=full_ids,
        attention_mask=attn,
        image_grid_hws=image_grid_hws,
        image_flags=None,
        return_dict=True,
    )
    logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
    token_logp = logp.gather(-1, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    prompt_len = int(prompt_ids.size(1))
    t = torch.arange(token_logp.size(1), device=device)
    mask = (t + 1 >= prompt_len).to(dtype=token_logp.dtype)
    if float(mask.sum()) == 0:
        return torch.zeros(
            (), device=device, dtype=token_logp.dtype, requires_grad=True
        )
    return (token_logp * mask).sum() / mask.sum().clamp_min(1)


def _prune_checkpoints(output_dir: Path, keep: int = 3):
    found = []
    for p in output_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-", 1)[1])
        except ValueError:
            continue
        found.append((step, p))
    found.sort()
    for _, p in found[:-keep]:
        shutil.rmtree(p)
        print(f"[grpo] pruned {p}", flush=True)


def _save_ckpt(worker, output_dir: Path, step: int):
    dest = output_dir / f"checkpoint-{step}"
    dest.mkdir(parents=True, exist_ok=True)
    worker.model.save_pretrained(dest)
    print(f"[grpo] saved {dest}", flush=True)
    _prune_checkpoints(output_dir, keep=3)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs))


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tiles = freeze_check.dedupe_by_image(freeze_check.load_samples(args.ann))
    if not tiles:
        raise RuntimeError(f"no samples in {args.ann}")
    print(f"[grpo] {len(tiles)} unique tiles from {args.ann}", flush=True)

    worker = rollout.make_worker(args.model_path, device=args.device)
    model = worker.model
    lora_params = _prepare_policy(model)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    model.eval()

    rng = Random(args.seed)
    order = list(range(len(tiles)))
    rng.shuffle(order)
    cursor = 0
    n_skip = 0

    for step in range(1, args.max_steps + 1):
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        row = tiles[order[cursor]]
        cursor += 1

        gt = freeze_check.parse_gt(row)
        img = freeze_check.load_image(args, row["image"])
        prompt = row["conversations"][0]["value"]
        model.eval()
        samples = rollout.sample_tile(
            worker,
            img,
            prompt,
            n=args.n,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        scored = [
            reward.compute_reward(s["answer"], gt, tile_size=TILE_SIZE) for s in samples
        ]
        rewards = [float(r["r_main"]) for r in scored]
        parse_ok_frac = sum(1.0 for r in scored if r["parse_ok"]) / max(len(scored), 1)
        mean_r = float(statistics.fmean(rewards)) if rewards else float("nan")
        std_r = _std(rewards)
        adv, skip = grpo_advantages(rewards)

        if skip:
            n_skip += 1
            print(
                f"step={step} zero_adv loss=nan mean_r={mean_r:.4f} std_r={std_r:.4f} "
                f"n_skip={n_skip} parse_ok_frac={parse_ok_frac:.3f}",
                flush=True,
            )
        else:
            model.train()
            model.vision_model.eval()
            model.mlp1.eval()
            seq_logps = [
                completion_logprob(
                    model,
                    worker.processor,
                    img,
                    prompt,
                    s["sequences"],
                    worker.device,
                    worker.dtype,
                    worker,
                )
                for s in samples
            ]
            seq_logp = torch.stack(seq_logps)
            adv_t = torch.tensor(adv, device=seq_logp.device, dtype=seq_logp.dtype)
            loss = -(adv_t.detach() * seq_logp).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            model.eval()
            print(
                f"step={step} loss={float(loss.detach().cpu()):.6f} "
                f"mean_r={mean_r:.4f} std_r={std_r:.4f} "
                f"n_skip={n_skip} parse_ok_frac={parse_ok_frac:.3f}",
                flush=True,
            )

        if step % args.save_steps == 0 or step == args.max_steps:
            _save_ckpt(worker, output_dir, step)

    return 0


if __name__ == "__main__":
    sys.exit(main())
