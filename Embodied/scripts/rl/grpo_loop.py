#!/usr/bin/env python
"""RL Phase B v2：group-normalized on-policy policy gradient + reference KL（HBB small）。

明确不是多 epoch PPO：一次采样、一次更新的 GRPO 型策略梯度。
采样分布锁在 rollout.RL_GENERATION_KW（slow / top_p=1 / top_k=0 / rep=1），
temperature 1.0；每个 group 只消费一次。
数据：RL-single JSONL（单 tile × 单类 prompt × 完整 GT），(image, prompt) 唯一；
不再跨图 dedupe。奖励 reward.py v3（一对一 soft 匹配 + small recall，输出上限 256）。

策略/参考：PEFT LoRA 双 adapter——policy = checkpoint 既有 SFT adapter（默认名
"default"，非默认时从 peft_config 读唯一现有名），reference = 冻结副本（字面名
"reference"，初始化为 policy 权重拷贝）。reference 分支 torch.no_grad() 前向。
loss = -(A·Σlogp − β·ΣKL) / 512（固定分母，不按 completion 长度归一），组内再 /G；
KL = exp(clamp(ref−pol, ±20)) − (ref−pol) − 1（k3 估计，逐 token）。

resume：checkpoint 同时含推理模型（已滤除 reference key）与 trainer_state.pt
（optimizer、global step、shuffle order/cursor、RNG、skip 计数、reference adapter）。
缺 trainer_state.pt 直接报错，不允许静默重置。
"""
from __future__ import annotations

import argparse
import json
import math
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
SMALL_AREA = 32.0 ** 2
REFERENCE_ADAPTER = "reference"  # 字面名；policy 名从 checkpoint peft_config 读取


def n_small_gt(gt: list) -> int:
    n = 0
    for _lab, x1, y1, x2, y2 in gt:
        if (x2 - x1) * (y2 - y1) < SMALL_AREA:
            n += 1
    return n


def grpo_advantages(rewards: list[float], min_std: float = 0.02) -> tuple[list[float], bool]:
    """组内标准化优势。std < min_std → 全 0 且 skip=True（本 step 不 optimizer.step，
    不把接近零的噪声放大为单位优势）。"""
    if len(rewards) == 0:
        return [], True
    mean = statistics.fmean(rewards)
    std = statistics.pstdev(rewards)
    if std < min_std:
        return [0.0] * len(rewards), True
    return [(r - mean) / std for r in rewards], False


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model-path",
        default="work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1",
    )
    p.add_argument(
        "--ann",
        default="data/dota_v1_hbb_448_rl_v2/annotations/DOTA-v1.0_train_rl_single_hbb_448.jsonl",
    )
    p.add_argument("--data-root", default="data/dota_v1_hbb_448_rl_v2")
    p.add_argument("--output-dir", default="work_dirs/rl_grpo_hbb_dense_v2")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="从该 checkpoint 恢复 policy/optimizer/order/RNG；缺 trainer_state.pt 报错")
    p.add_argument("--n", type=int, default=16, help="组大小 G")
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-output-boxes", type=int, default=reward.MAX_BOXES)
    p.add_argument("--min-reward-std", type=float, default=0.02,
                   help="组 reward std 低于此值 → skip（无单位优势放大噪声）")
    p.add_argument("--kl-beta", type=float, default=0.02)
    p.add_argument("--loss-token-normalizer", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--min-small-gt",
        type=int,
        default=0,
        help="只保留 COCO-small GT 数 ≥ 该值的样本（0=不过滤；对单类完整 GT 逐行计）",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# 数据（RL-single：按 (image, prompt) 唯一，不跨图 dedupe）
# --------------------------------------------------------------------------- #

def load_rl_samples(args) -> list:
    rows = freeze_check.load_samples(args.ann)
    seen: dict = {}
    for row in rows:
        key = (row["image"], freeze_check.sample_prompt(row))
        if key in seen:
            raise ValueError(f"duplicate (image, prompt) row: {key[0]} :: {key[1][:80]}...")
        seen[key] = row
    population = list(seen.values())
    print(f"[grpo] {len(population)} unique (image, prompt) samples from {args.ann}", flush=True)
    if args.min_small_gt > 0:
        kept = []
        for row in population:
            gt = freeze_check.parse_gt(row)
            if n_small_gt(gt) >= args.min_small_gt:
                kept.append(row)
        print(f"[grpo] smallGT>={args.min_small_gt}: {len(kept)}/{len(population)} samples",
              flush=True)
        population = kept
        if not population:
            raise RuntimeError(f"no samples with smallGT>={args.min_small_gt}")
    return population


def log_population_stats(samples: list) -> None:
    gts = [freeze_check.parse_gt(row) for row in samples]
    counts = [len(g) for g in gts]
    smalls = [n_small_gt(g) for g in gts]
    class_counts: dict = {}
    for g in gts:
        if g:
            class_counts[g[0][0]] = class_counts.get(g[0][0], 0) + 1

    def pct(a, q):
        s = sorted(a)
        return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))] if s else 0

    print(f"[grpo] selected={len(samples)} classes={json.dumps(class_counts, ensure_ascii=False)}",
          flush=True)
    print(f"[grpo] gt_boxes p50={pct(counts, 0.5)} p90={pct(counts, 0.9)} max={max(counts)}; "
          f"small_gt p50={pct(smalls, 0.5)} p90={pct(smalls, 0.9)} max={max(smalls)}", flush=True)


# --------------------------------------------------------------------------- #
# policy / reference adapter（PEFT 0.12）
# --------------------------------------------------------------------------- #

def _collect_lora_params(model):
    return [(n, p) for n, p in model.named_parameters() if "lora_" in n]


def _peft_model(model):
    try:
        from peft import PeftModel
        if isinstance(model.language_model, PeftModel):
            return model.language_model
    except ImportError:
        pass
    return None


def _prepare_policy(model) -> tuple[list, str]:
    """确保 policy LoRA 存在，创建冻结 reference 副本，恢复 requires-grad。

    返回 (trainable lora params, policy adapter name)。
    """
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    import copy

    peft_model = _peft_model(model)
    if peft_model is None:
        print(f"[grpo] language_model type={type(model.language_model)!r} → wrap_llm_lora(64)",
              flush=True)
        model.wrap_llm_lora(r=64, lora_alpha=128)
        peft_model = _peft_model(model)
        if peft_model is None:
            raise RuntimeError("language_model is not a PeftModel after wrap_llm_lora")

    existing = list(peft_model.peft_config.keys())
    if len(existing) == 1:
        policy_name = existing[0]
    elif len(existing) == 0:
        raise RuntimeError("PeftModel has no adapters after load/wrap")
    else:
        raise RuntimeError(f"multiple existing adapters {existing}; refusing to guess policy")
    if REFERENCE_ADAPTER in existing:
        raise RuntimeError(f"adapter {REFERENCE_ADAPTER!r} already exists in checkpoint")
    ref_cfg = copy.deepcopy(peft_model.peft_config[policy_name])
    peft_model.add_adapter(REFERENCE_ADAPTER, ref_cfg)
    ref_state = get_peft_model_state_dict(peft_model, adapter_name=policy_name)
    set_peft_model_state_dict(peft_model, ref_state, adapter_name=REFERENCE_ADAPTER)
    # add_adapter 建出的新 adapter 继承 base dtype（bf16），而 checkpoint 的 policy
    # adapter 常为 fp32；dtype 不一致会让 ref/pol 前向舍入不同（fresh run KL≠0）。
    # torch.equal 跨 dtype 按值比较查不出这种差异，必须显式统一到 policy dtype。
    pol_named = {n: p for n, p in model.named_parameters() if "lora_" in n}
    for n, p in pol_named.items():
        if f".{REFERENCE_ADAPTER}." not in n:
            continue
        twin = n.replace(f".{REFERENCE_ADAPTER}.", f".{policy_name}.")
        src = pol_named.get(twin)
        if src is not None and p.dtype != src.dtype:
            p.data = p.data.to(src.dtype)
    peft_model.set_adapter(policy_name)

    # requires-grad：只有 policy LoRA 可训练；reference / vision / MLP / base 全冻结
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = f".{policy_name}." in n
        else:
            p.requires_grad = False
    model.vision_model.requires_grad_(False)
    model.mlp1.requires_grad_(False)
    model.vision_model.eval()
    model.mlp1.eval()

    lora_params = [p for n, p in _collect_lora_params(model) if p.requires_grad]
    if not lora_params:
        raise RuntimeError("no trainable policy lora parameters")
    n_ref = sum(1 for n, _p in _collect_lora_params(model) if f".{REFERENCE_ADAPTER}." in n)
    print(f"[grpo] policy adapter={policy_name!r} trainable params={sum(p.numel() for p in lora_params)}; "
          f"reference adapter frozen tensors={n_ref}", flush=True)
    return lora_params, policy_name


# --------------------------------------------------------------------------- #
# 前向：每 group 一次 prompt 上下文；逐 completion 的 token log-probs
# --------------------------------------------------------------------------- #

def prepare_prompt_context(model, processor, image, prompt, device, dtype, worker) -> dict:
    """每个 group 只计算一次 prompt IDs 与冻结 MoonViT/MLP 视觉特征。"""
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
    with torch.no_grad():
        vit_embeds = model.extract_feature(pixel_values, image_grid_hws)
        if image_grid_hws is not None:
            vit_embeds = torch.cat(vit_embeds, dim=0)
            vit_embeds = model.mlp1(vit_embeds)
    return {"prompt_ids": prompt_ids, "prompt_len": int(prompt_ids.size(1)),
            "vit_embeds": vit_embeds}


def completion_token_logprobs(model, context: dict, completion_ids, adapter_name: str,
                              device) -> torch.Tensor:
    """completion 每个 token 的 log-prob [T]（completion 区域，不含 prompt）。

    reference 分支由调用方包 torch.no_grad()。adapter 切换在本函数内 set_adapter；
    调用方按 policy → reference → policy 的顺序成对使用并在组末留在 policy。
    """
    comp = completion_ids.view(-1).to(device)
    prompt_ids = context["prompt_ids"]
    if comp.numel() == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)

    full_ids = torch.cat([prompt_ids, comp.unsqueeze(0)], dim=1)
    attn = torch.ones(full_ids.shape, dtype=torch.long, device=device)
    vit_embeds = context["vit_embeds"]

    # eval 模式 + clone 注入（100k Qwen2 的 self.training 分支不可用）；
    # enable_input_require_grads 让 embedding 成 leaf，image_processing inplace 会炸 → clone。
    qwen = model.language_model
    while not hasattr(qwen, "image_processing") and hasattr(qwen, "model"):
        qwen = qwen.model
    orig_ip = qwen.image_processing

    def _cloned_image_processing(input_ids, visual_features, image_token_index):
        input_embeds = qwen.get_input_embeddings()(input_ids)
        if visual_features is None:
            return input_embeds
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C).clone()
        selected = input_ids.reshape(B * N) == image_token_index
        n = int(selected.sum())
        if n:
            input_embeds[selected] = visual_features.reshape(-1, C).to(
                dtype=input_embeds.dtype, device=input_embeds.device
            )[:n]
        return input_embeds.reshape(B, N, C)

    qwen.image_processing = _cloned_image_processing
    peft_model = model.language_model
    try:
        peft_model.set_adapter(adapter_name)
        out = model.language_model(
            input_ids=full_ids,
            attention_mask=attn,
            visual_features=vit_embeds,
            image_token_index=model.config.image_token_index,
            use_cache=False,
            return_dict=True,
        )
    finally:
        qwen.image_processing = orig_ip
    if isinstance(out, tuple):
        out = out[0]
    logp = F.log_softmax(out.logits[:, :-1, :], dim=-1)
    token_logp = logp.gather(-1, full_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    prompt_len = context["prompt_len"]
    t = torch.arange(token_logp.size(1), device=device)
    mask = (t + 1 >= prompt_len).to(dtype=token_logp.dtype)
    return token_logp * mask


def sequence_policy_loss(policy_logp: torch.Tensor, ref_logp: torch.Tensor,
                         advantage: float, kl_beta: float,
                         token_normalizer: float) -> tuple[torch.Tensor, torch.Tensor]:
    """-(A·Σlogp − β·ΣKL) / normalizer 与 mean KL（float32，k3 估计）。

    分母固定 token_normalizer（默认 512）而非各 completion 实际长度：只改全局梯度
    尺度，不系统性降低长多框轨迹的权重。
    """
    policy_logp = policy_logp.float()
    ref_logp = ref_logp.float()
    delta = (ref_logp - policy_logp).clamp(-20.0, 20.0)
    kl = torch.exp(delta) - delta - 1.0
    loss = -(advantage * policy_logp.sum() - kl_beta * kl.sum()) / float(token_normalizer)
    return loss, kl.mean().detach()


# --------------------------------------------------------------------------- #
# checkpoint：外层 state dict 滤 reference；trainer_state.pt 支持精确 resume
# --------------------------------------------------------------------------- #

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


def _save_ckpt(worker, output_dir: Path, step: int, optimizer, policy_name: str,
               extra_state: dict):
    from peft import get_peft_model_state_dict

    peft_model = worker.model.language_model
    peft_model.set_adapter(policy_name)  # 保存前强制切回 policy
    dest = output_dir / f"checkpoint-{step}"
    dest.mkdir(parents=True, exist_ok=True)
    full_sd = worker.model.state_dict()
    filtered = {k: v for k, v in full_sd.items() if f".{REFERENCE_ADAPTER}." not in k}
    n_dropped = len(full_sd) - len(filtered)
    worker.model.save_pretrained(dest, state_dict=filtered)
    worker.tokenizer.save_pretrained(dest)   # resume/eval 需要从 checkpoint 直接加载
    worker.processor.save_pretrained(dest)
    state = dict(extra_state)
    state["global_step"] = step
    state["policy_adapter"] = policy_name
    state["reference_adapter_state"] = get_peft_model_state_dict(
        peft_model, adapter_name=REFERENCE_ADAPTER)
    torch.save(state, dest / "trainer_state.pt")
    print(f"[grpo] saved {dest} (dropped {n_dropped} reference tensors, "
          f"trainer_state step={step})", flush=True)
    _prune_checkpoints(output_dir, keep=3)


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    return float(statistics.pstdev(xs))


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, round(q * len(s)) - 1))] if s else float("nan")


def main(argv=None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    samples = load_rl_samples(args)
    log_population_stats(samples)

    resume_state = None
    if args.resume_from_checkpoint:
        ckpt = Path(args.resume_from_checkpoint)
        state_path = ckpt / "trainer_state.pt"
        if not state_path.exists():
            raise RuntimeError(
                f"{state_path} missing: resume requires trainer_state.pt; "
                "refusing to silently reset optimizer/order")
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        print(f"[grpo] resuming from {ckpt} (global step {resume_state['global_step']})",
              flush=True)
        worker = rollout.make_worker(str(ckpt), device=args.device)
    else:
        worker = rollout.make_worker(args.model_path, device=args.device)
    model = worker.model
    lora_params, policy_name = _prepare_policy(model)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    model.eval()

    if resume_state is not None:
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(model.language_model,
                                  resume_state["reference_adapter_state"],
                                  adapter_name=REFERENCE_ADAPTER)
        optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state.get("policy_adapter") != policy_name:
            raise RuntimeError(
                f"policy adapter mismatch: checkpoint {resume_state['policy_adapter']!r} "
                f"vs loaded {policy_name!r}")
        rng = Random(args.seed)
        rng.setstate(resume_state["rng_state"])
        order = resume_state["order"]
        cursor = resume_state["cursor"]
        n_skip = resume_state["n_skip"]
        start_step = resume_state["global_step"]
        torch.set_rng_state(resume_state["torch_rng_state"])
        if torch.cuda.is_available() and resume_state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(resume_state["cuda_rng_state"])
        print(f"[grpo] restored optimizer/order({len(order)})/cursor={cursor}/"
              f"n_skip={n_skip}; next global step = {start_step + 1}", flush=True)
    else:
        rng = Random(args.seed)
        order = list(range(len(samples)))
        rng.shuffle(order)
        cursor = 0
        n_skip = 0
        start_step = 0

    max_logp_diff_printed = False
    for step in range(start_step + 1, args.max_steps + 1):
        if cursor >= len(order):
            rng.shuffle(order)
            cursor = 0
        row = samples[order[cursor]]
        cursor += 1

        gt = freeze_check.parse_gt(row)
        img = freeze_check.load_image(args, row["image"])
        prompt = freeze_check.sample_prompt(row)
        model.eval()
        samples_g = rollout.sample_tile(
            worker,
            img,
            prompt,
            n=args.n,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        scored = [
            reward.compute_reward(s["answer"], gt, tile_size=TILE_SIZE) for s in samples_g
        ]
        rewards = [float(r["r_main"]) for r in scored]
        parse_ok_frac = sum(1.0 for r in scored if r["parse_ok"]) / max(len(scored), 1)
        n_preds = [int(r["n_pred"]) for r in scored]
        comp_tokens = [int(s["sequences"].numel()) for s in samples_g]
        mean_r = float(statistics.fmean(rewards)) if rewards else float("nan")
        std_r = _std(rewards)
        adv, skip = grpo_advantages(rewards, min_std=args.min_reward_std)

        if skip:
            n_skip += 1
            print(
                f"step={step} skip=zero_adv(std={std_r:.4f}<{args.min_reward_std}) "
                f"loss=nan mean_r={mean_r:.4f} f1_soft=nan small_recall=nan mean_kl=nan "
                f"n_pred={_pct(n_preds, 0.5):.0f} tok={_pct(comp_tokens, 0.5):.0f} "
                f"n_skip={n_skip} parse_ok_frac={parse_ok_frac:.3f}",
                flush=True,
            )
        else:
            # 一次采样、一次更新；eval 模式（见 completion_token_logprobs 注释）。
            model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            context = prepare_prompt_context(
                model, worker.processor, img, prompt, worker.device, worker.dtype, worker
            )
            optimizer.zero_grad(set_to_none=True)
            g = max(len(samples_g), 1)
            loss_acc = 0.0
            kl_acc = 0.0
            for s, a in zip(samples_g, adv):
                with torch.no_grad():
                    ref_logp = completion_token_logprobs(
                        model, context, s["sequences"], REFERENCE_ADAPTER, worker.device
                    )
                pol_logp = completion_token_logprobs(
                    model, context, s["sequences"], policy_name, worker.device
                )
                if not max_logp_diff_printed and ref_logp.numel() and pol_logp.numel():
                    d = float((pol_logp - ref_logp).abs().max())
                    print(f"[grpo] first group max|pol-ref| token logp diff = {d:.2e} "
                          f"(fresh run 必须≈0：reference 为 policy 拷贝)", flush=True)
                    max_logp_diff_printed = True
                li, kl_i = sequence_policy_loss(
                    pol_logp, ref_logp, float(a), args.kl_beta, args.loss_token_normalizer
                )
                (li / g).backward()
                loss_acc += float(li.detach().cpu())
                kl_acc += float(kl_i.cpu())
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            model.language_model.set_adapter(policy_name)  # 组末留在 policy
            f1s = float(statistics.fmean([float(r["f1_soft"]) for r in scored]))
            rs = float(statistics.fmean([float(r["small_recall_soft"]) for r in scored]))
            print(
                f"step={step} loss={loss_acc:.6f} mean_r={mean_r:.4f} std_r={std_r:.4f} "
                f"f1_soft={f1s:.3f} small_recall={rs:.3f} mean_kl={kl_acc / g:.4f} "
                f"n_pred(mean/max)={statistics.fmean(n_preds):.1f}/{max(n_preds)} "
                f"tok(mean/max)={statistics.fmean(comp_tokens):.0f}/{max(comp_tokens)} "
                f"n_skip={n_skip} parse_ok_frac={parse_ok_frac:.3f}",
                flush=True,
            )

        if step % args.save_steps == 0 or step == args.max_steps:
            _save_ckpt(
                worker, output_dir, step, optimizer, policy_name,
                extra_state={
                    "order": order,
                    "cursor": cursor,
                    "n_skip": n_skip,
                    "rng_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (torch.cuda.get_rng_state()
                                       if torch.cuda.is_available() else None),
                },
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
