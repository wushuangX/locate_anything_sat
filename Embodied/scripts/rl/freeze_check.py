#!/usr/bin/env python
"""RL Phase A freeze-check gate v2（dryrun / timing / gate / summarize）。

数据：RL-single JSONL（converter --emit-rl-single-class 输出），每行 = 单 tile ×
单类 prompt × 完整 GT；按 (image, human prompt) 唯一。跨图 dedupe（dedupe_by_image）
已删除——旧协议按 image 取首行会拿到与 eval 句式不符的多类 T1 prompt。

门（同一 prompt 组内是否排对候选，设计定值不许调）：
  gate_pairwise  : 组内 anchor 不同的候选对中，奖励差与 anchor 差同号的比例
                   （先组内求值再宏平均）>= 0.70
  gate_group_rho : reward/anchor 都非恒定组的 Spearman 中位数 >= 0.60
  gate_signal    : reward 标准差 >= 0.02 的组占比 >= 0.70
  gate_timing    : mean reward_seconds <= 0.25 × mean gen_seconds

anchor = 0.5 × hard_F1@IoU0.5 + 0.5 × hard_recall_small（与 dense eval 同构：
IoU 降序贪心、一对一、IoU>=0.5；small = COCO 面积 <32^2 px）。
全局 Spearman / reward 分位 / parse fail / 平均预测框数 / small recall 仅诊断。
只有四个门全部通过才能启动 policy update；summarize 任一门失败 exit 1，唯一合法
后续是修 reward.py 后 `--stage summarize --rescore`（从 answer 列重算，不重跑生成）。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Embodied 根（worker）
sys.path.insert(0, str(Path(__file__).resolve().parent))  # scripts/rl（reward/rollout）

import reward  # noqa: E402

TILE_SIZE = 448  # DOTA HBB tile 尺寸（token→像素换算基准）
SMALL_AREA = 32.0 ** 2

PAIRWISE_THR = 0.70
GROUP_SPEARMAN_THR = 0.60
SIGNAL_FRACTION_THR = 0.70
SIGNAL_STD_THR = 0.02
TIMING_RATIO_THR = 0.25
GATE_WALLCLOCK_BUDGET_H = 12.0  # timing 阶段的预估上限（信息门）
FALLBACK_GEN_SECONDS = 10.0  # 单样本生成均值超此值 → 触发预决策回退 --samples 384 --n 12


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True,
                   choices=["dryrun", "timing", "gate", "summarize"])
    p.add_argument("--model-path",
                   default="work_dirs/dota_geom_lora_lda_2gpu_4k_100k_run1")
    p.add_argument("--ann",
                   default="data/dota_v1_hbb_448_rl_v2/annotations/DOTA-v1.0_train_rl_single_hbb_448.jsonl")
    p.add_argument("--data-root", default="data/dota_v1_hbb_448_rl_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="work_dirs/rl_freeze_v2")
    p.add_argument("--samples", type=int, default=512,
                   help="抽查的 RL-single 样本数（(image,prompt) 行）")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--min-small-gt", type=int, default=10,
                   help="只保留 COCO-small GT 数 >= 该值的样本（按该单类行计）")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--max-output-boxes", type=int, default=reward.MAX_BOXES)
    p.add_argument("--rescore", action="store_true",
                   help="仅 summarize：从 per_sample 的 answer 列重算奖励，不重跑生成")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #

def load_samples(ann_path: str) -> list:
    samples = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def sample_prompt(row: dict) -> str:
    return row["conversations"][0]["value"]


def sample_id(row: dict) -> str:
    return row["image"] + "\n" + sample_prompt(row)


def select_samples(args) -> list:
    samples = load_samples(args.ann)
    print(f"[data] loaded {len(samples)} rows from {args.ann}")
    print("[data] FIRST SAMPLE (字段核对, AGENTS §4.1):")
    print(json.dumps(samples[0], ensure_ascii=False, indent=2))
    seen: dict = {}
    for row in samples:
        key = (row["image"], sample_prompt(row))
        if key in seen:
            raise ValueError(
                f"duplicate (image, prompt) row: {key[0]} :: {key[1][:80]}... "
                "RL-single JSONL must have unique (image, prompt)")
        seen[key] = row
    population = list(seen.values())
    print(f"[data] {len(population)} unique (image, prompt) samples")
    if args.min_small_gt > 0:
        filtered = []
        for row in population:
            gt = parse_gt(row)
            n_small = sum(1 for g in gt if (g[3] - g[1]) * (g[4] - g[2]) < SMALL_AREA)
            if n_small >= args.min_small_gt:
                filtered.append(row)
        print(f"[data] smallGT>={args.min_small_gt}: {len(filtered)}/{len(population)} samples")
        population = filtered
    k = min(args.samples, len(population))
    chosen = random.Random(args.seed).sample(population, k)
    return chosen


def shard_split(items: list, shard: int, num_shards: int) -> list:
    return items[shard::num_shards]


def parse_gt(row: dict) -> list:
    # GT 是完整单类标注，上限独立于模型输出（reward.GT_MAX_BOXES）
    return reward.parse_answer(sample_prompt_gt(row), TILE_SIZE, max_boxes=reward.GT_MAX_BOXES)["boxes"]


def sample_prompt_gt(row: dict) -> str:
    return row["conversations"][1]["value"]


def load_image(args, rel_path: str):
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # 防 DecompressionBomb 保护拦截
    return Image.open(Path(args.data_root) / rel_path).convert("RGB")


# --------------------------------------------------------------------------- #
# 逐样本评测（生成 + 奖励 + hard anchor）
# --------------------------------------------------------------------------- #

def score_one(answer: str, gt: list, gen_seconds: float, max_output_boxes: int = reward.MAX_BOXES) -> dict:
    t0 = time.perf_counter()
    r = reward.compute_reward(answer, gt, tile_size=TILE_SIZE)
    pred_boxes = reward.parse_answer(answer, TILE_SIZE, max_boxes=max_output_boxes)["boxes"]
    matches = reward.match_at_iou(pred_boxes, gt, 0.5)
    tp = len(matches)
    n_pred, n_gt = len(pred_boxes), len(gt)
    hard_f1_all = 2.0 * tp / (n_pred + n_gt) if (n_pred + n_gt) else 0.0
    n_small_gt = sum(1 for g in gt if (g[3] - g[1]) * (g[4] - g[2]) < SMALL_AREA)
    if n_small_gt:
        hard_recall_small = sum(1 for _i, j, _s in matches
                                if (gt[j][3] - gt[j][1]) * (gt[j][4] - gt[j][2]) < SMALL_AREA) / n_small_gt
    else:
        hard_recall_small = 0.0
    anchor = 0.5 * hard_f1_all + 0.5 * hard_recall_small
    reward_seconds = time.perf_counter() - t0
    return {
        "answer": answer,
        "r_main": r["r_main"],
        "f1_soft": r["f1_soft"],
        "small_recall_soft": r["small_recall_soft"],
        "hard_f1_all": hard_f1_all,
        "hard_recall_small": hard_recall_small,
        "anchor": anchor,
        "n_raw": r["n_raw"],
        "n_pred": n_pred,
        "n_gt": n_gt,
        "n_small_gt": n_small_gt,
        "is_none": r["is_none"],
        "parse_ok": r["parse_ok"],
        "gen_seconds": gen_seconds,
        "reward_seconds": reward_seconds,
    }


def trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 3] + "..."


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def run_dryrun(args) -> int:
    import rollout

    samples = select_samples(args)[:4]
    worker = rollout.make_worker(args.model_path)
    n = 2
    rows = []
    for row in samples:
        gt = parse_gt(row)
        img = load_image(args, row["image"])
        prompt = sample_prompt(row)
        for s in rollout.sample_tile(worker, img, prompt, n, max_new_tokens=args.max_new_tokens):
            rec = score_one(s["answer"], gt, s["gen_seconds"], args.max_output_boxes)
            rec.update(image=row["image"], prompt=prompt, sample_id=sample_id(row), shard=args.shard)
            rows.append(rec)
            print(f"[dryrun] {Path(row['image']).name} r_main={rec['r_main']:+.3f} "
                  f"anchor={rec['anchor']:.3f} parse_ok={rec['parse_ok']} "
                  f"n_pred={rec['n_pred']} none={rec['is_none']}")
            print(f"    prompt: {trunc(prompt, 100)}")
            print(f"    answer: {trunc(rec['answer'], 140)}")
    rate = sum(r["parse_ok"] for r in rows) / len(rows)
    print(f"[dryrun] parse_ok rate = {rate:.2f} ({sum(r['parse_ok'] for r in rows)}/{len(rows)}); "
          f"放行阈值 >= 0.8")
    return 0 if rate >= 0.8 else 1


def run_timing(args) -> int:
    import rollout

    samples = select_samples(args)[:32]
    worker = rollout.make_worker(args.model_path)
    gen_s, reward_s = [], []
    t_start = time.perf_counter()
    for i, row in enumerate(samples):
        gt = parse_gt(row)
        img = load_image(args, row["image"])
        for s in rollout.sample_tile(worker, img, sample_prompt(row), args.n, max_new_tokens=args.max_new_tokens):
            rec = score_one(s["answer"], gt, s["gen_seconds"], args.max_output_boxes)
            gen_s.append(rec["gen_seconds"])
            reward_s.append(rec["reward_seconds"])
        done = i + 1
        el = time.perf_counter() - t_start
        print(f"[timing] {done}/{len(samples)} samples, elapsed {el / 60:.1f} min, "
              f"ETA {(el / done) * (len(samples) - done) / 60:.1f} min", flush=True)

    gen_s.sort()
    reward_s.sort()

    def pct(a, q):
        return a[min(len(a) - 1, max(0, math.ceil(q * len(a)) - 1))]

    mean_gen = sum(gen_s) / len(gen_s)
    mean_reward = sum(reward_s) / len(reward_s)
    total_samples = args.samples * args.n
    wall_h = (mean_gen + mean_reward) * total_samples / args.num_shards / 3600.0
    print("[timing] ===== 计时表 =====")
    print(f"  gen_seconds  : mean={mean_gen:.3f} median={pct(gen_s, 0.5):.3f} "
          f"p90={pct(gen_s, 0.9):.3f} max={gen_s[-1]:.3f}")
    print(f"  reward_seconds: mean={mean_reward:.4f} median={pct(reward_s, 0.5):.4f} "
          f"max={reward_s[-1]:.4f}")
    print(f"  reward/gen ratio = {mean_reward / mean_gen:.4f} (gate_timing 阈值 0.25)")
    print(f"  gate 预估（{args.samples} samples × {args.n} × {args.num_shards} 卡）= {wall_h:.2f} h "
          f"(预算 {GATE_WALLCLOCK_BUDGET_H:.0f} h)")
    fallback = mean_gen > FALLBACK_GEN_SECONDS
    print(f"  单样本生成均值 {'>' if fallback else '<='} {FALLBACK_GEN_SECONDS}s → "
          + ("触发预决策回退：--samples 384 --n 12（门阈值不变）" if fallback else "无需回退"))
    return 1 if fallback else 0


def run_gate(args) -> int:
    import rollout

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = out_dir / f"per_sample_shard{args.shard}.jsonl"

    done = set()
    if per_sample_path.exists():
        with open(per_sample_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done.add((r["sample_id"], r["k"]))
        print(f"[gate] resume: {len(done)} (sample_id,k) already done in {per_sample_path}")

    samples = shard_split(select_samples(args), args.shard, args.num_shards)
    print(f"[gate] shard {args.shard}/{args.num_shards}: {len(samples)} samples × n={args.n}")
    worker = rollout.make_worker(args.model_path)

    t_start = time.perf_counter()
    n_done = 0
    with open(per_sample_path, "a", encoding="utf-8") as fout:
        for i, row in enumerate(samples):
            sid = sample_id(row)
            need = [k for k in range(args.n) if (sid, k) not in done]
            if not need:
                continue
            gt = parse_gt(row)
            img = load_image(args, row["image"])
            prompt = sample_prompt(row)
            for k in need:
                s = rollout.sample_tile(worker, img, prompt, 1, max_new_tokens=args.max_new_tokens)[0]
                rec = score_one(s["answer"], gt, s["gen_seconds"], args.max_output_boxes)
                rec.update(image=row["image"], prompt=prompt, sample_id=sid, shard=args.shard, k=k)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                n_done += 1
            n_done_samples = i + 1
            if n_done_samples % 10 == 0 or n_done_samples == len(samples):
                el = time.perf_counter() - t_start
                eta = el / max(n_done, 1) * (len(samples) * args.n - (len(done) + n_done)) \
                    if n_done else 0.0
                print(f"[gate] shard {args.shard}: {n_done_samples}/{len(samples)} samples, "
                      f"{len(done) + n_done} rollouts, elapsed {el / 60:.1f} min, "
                      f"ETA {eta / 60:.1f} min", flush=True)
    print(f"[gate] shard {args.shard} finished: wrote {n_done} new rollouts to {per_sample_path}")
    return 0


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #

def _ranks(v: list) -> list:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 平均秩（1-based）
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return r

def spearman(x: list, y: list) -> float:
    try:
        from scipy.stats import spearmanr  # type: ignore

        return float(spearmanr(x, y).correlation)
    except Exception:
        rx, ry = _ranks(x), _ranks(y)
        mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
        cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        vx = sum((a - mx) ** 2 for a in rx)
        vy = sum((b - my) ** 2 for b in ry)
        if vx <= 0 or vy <= 0:
            return float("nan")
        return cov / math.sqrt(vx * vy)


def _pct(a: list, q: float) -> float:
    a = sorted(a)
    return a[min(len(a) - 1, max(0, math.ceil(q * len(a)) - 1))]


def _mean(a: list) -> float:
    return sum(a) / len(a) if a else float("nan")


def _std(a: list) -> float:
    if len(a) < 2:
        return 0.0
    m = _mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / (len(a) - 1))


def load_per_sample_rows(out_dir: Path) -> list:
    rows = []
    paths = sorted(out_dir.glob("per_sample_shard*.jsonl"))
    if not paths:
        sys.exit(f"[summarize] no per_sample_shard*.jsonl under {out_dir}")
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def rescore_rows(args, rows: list) -> list:
    """从 answer 列重算奖励字段（修 reward.py 后使用；不重跑生成）。GT 按 (image,prompt) join。"""
    gt_map = {}
    for row in load_samples(args.ann):
        gt_map[(row["image"], sample_prompt(row))] = parse_gt(row)
    for r in rows:
        gt = gt_map.get((r["image"], r["prompt"]), [])
        rec = score_one(r["answer"], gt, r["gen_seconds"], args.max_output_boxes)
        r.update({k: rec[k] for k in rec if k != "answer"})
    # 按 shard 分组回写原文件
    by_shard = {}
    for r in rows:
        by_shard.setdefault(r["shard"], []).append(r)
    for shard, rs in by_shard.items():
        p = Path(args.out_dir) / f"per_sample_shard{shard}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[summarize] rescored {len(rs)} rows -> {p}")
    return rows


def _group_concordance(pairs_reward: list, pairs_anchor: list) -> float:
    """组内 anchor 不同的候选对中，奖励差与 anchor 差同号的比例。无有效对 → nan。"""
    n_info = 0
    n_conc = 0
    for i in range(len(pairs_reward)):
        for j in range(i + 1, len(pairs_reward)):
            da = pairs_anchor[i] - pairs_anchor[j]
            if da == 0:
                continue
            n_info += 1
            if (pairs_reward[i] - pairs_reward[j]) * da > 0:
                n_conc += 1
    return n_conc / n_info if n_info else float("nan")


def run_summarize(args) -> int:
    out_dir = Path(args.out_dir)
    rows = load_per_sample_rows(out_dir)
    print(f"[summarize] loaded {len(rows)} rollouts from per_sample_shard*.jsonl")
    if args.rescore:
        rows = rescore_rows(args, rows)

    n_ok = sum(1 for r in rows if r["parse_ok"])

    # --- 分组：同一 (image, prompt) 内的候选 ---
    groups: dict = {}
    for r in rows:
        groups.setdefault(r["sample_id"], []).append(r)

    group_pairwise = []
    group_rhos = []
    n_signal = 0
    for members in groups.values():
        rs = [m["r_main"] for m in members]
        ancs = [m["anchor"] for m in members]
        conc = _group_concordance(rs, ancs)
        if math.isfinite(conc):
            group_pairwise.append(conc)
        if _std(rs) > 0 and _std(ancs) > 0:
            rho = spearman(rs, ancs)
            if math.isfinite(rho):
                group_rhos.append(rho)
        if _std(rs) >= SIGNAL_STD_THR:
            n_signal += 1
    n_groups = len(groups)

    # --- gate_pairwise ---
    pairwise = _mean(group_pairwise)
    gate_pairwise = math.isfinite(pairwise) and pairwise >= PAIRWISE_THR

    # --- gate_group_rho ---
    median_rho = float("nan")
    if group_rhos:
        srt = sorted(group_rhos)
        m = len(srt)
        median_rho = srt[m // 2] if m % 2 else 0.5 * (srt[m // 2 - 1] + srt[m // 2])
    gate_group_rho = math.isfinite(median_rho) and median_rho >= GROUP_SPEARMAN_THR

    # --- gate_signal ---
    signal_fraction = n_signal / n_groups if n_groups else float("nan")
    gate_signal = math.isfinite(signal_fraction) and signal_fraction >= SIGNAL_FRACTION_THR

    # --- gate_timing ---
    mean_reward_s = _mean([r["reward_seconds"] for r in rows])
    mean_gen_s = _mean([r["gen_seconds"] for r in rows])
    timing_ratio = mean_reward_s / mean_gen_s if mean_gen_s > 0 else float("inf")
    gate_timing = math.isfinite(timing_ratio) and timing_ratio <= TIMING_RATIO_THR

    # --- 诊断（不设门）---
    global_rho = (spearman([r["r_main"] for r in rows], [r["anchor"] for r in rows])
                  if len(rows) >= 2 else float("nan"))
    diag = {
        "n_total": len(rows),
        "n_parse_ok": n_ok,
        "n_groups": n_groups,
        "n_groups_informative_pairwise": len(group_pairwise),
        "n_groups_signal_spearman": len(group_rhos),
        "none_rate": _mean([1.0 if r["is_none"] else 0.0 for r in rows]),
        "parse_fail_rate": 1.0 - n_ok / len(rows) if rows else float("nan"),
        "r_main_p10": _pct([r["r_main"] for r in rows], 0.10),
        "r_main_p50": _pct([r["r_main"] for r in rows], 0.50),
        "r_main_p90": _pct([r["r_main"] for r in rows], 0.90),
        "mean_pred_boxes": _mean([r["n_pred"] for r in rows]),
        "mean_hard_recall_small": _mean([r["hard_recall_small"] for r in rows]),
        "spearman_global_reward_anchor": global_rho,
        "anchor": "0.5*hard_F1@0.5 + 0.5*hard_recall_small",
    }

    summary = {
        "model_path": args.model_path,
        "ann": args.ann,
        "samples": args.samples,
        "n": args.n,
        "seed": args.seed,
        "min_small_gt": args.min_small_gt,
        "max_new_tokens": args.max_new_tokens,
        "max_output_boxes": args.max_output_boxes,
        "rescored": bool(args.rescore),
        "gate_pairwise": {"value": pairwise, "threshold": PAIRWISE_THR,
                          "n_groups": len(group_pairwise), "pass": bool(gate_pairwise)},
        "gate_group_rho": {"value": median_rho, "threshold": GROUP_SPEARMAN_THR,
                           "n_groups": len(group_rhos), "pass": bool(gate_group_rho)},
        "gate_signal": {"value": signal_fraction, "threshold": SIGNAL_FRACTION_THR,
                        "signal_std_thr": SIGNAL_STD_THR, "pass": bool(gate_signal)},
        "gate_timing": {"mean_reward_seconds": mean_reward_s, "mean_gen_seconds": mean_gen_s,
                        "ratio": timing_ratio, "threshold": TIMING_RATIO_THR,
                        "pass": bool(gate_timing)},
        "diagnostics": diag,
        "all_pass": bool(gate_pairwise and gate_group_rho and gate_signal and gate_timing),
    }
    out_path = out_dir / "summary_gate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[summarize] ===== GATE =====")
    print(f"  gate_pairwise : {pairwise:.4f} over {len(group_pairwise)} groups "
          f"(thr >= {PAIRWISE_THR})  -> {'PASS' if gate_pairwise else 'FAIL'}")
    print(f"  gate_group_rho: median={median_rho:.4f} over {len(group_rhos)} groups "
          f"(thr >= {GROUP_SPEARMAN_THR}) -> {'PASS' if gate_group_rho else 'FAIL'}")
    print(f"  gate_signal   : {signal_fraction:.4f} of {n_groups} groups "
          f"(std >= {SIGNAL_STD_THR}; thr >= {SIGNAL_FRACTION_THR}) "
          f"-> {'PASS' if gate_signal else 'FAIL'}")
    print(f"  gate_timing   : reward/gen={timing_ratio:.4f} (thr <= {TIMING_RATIO_THR}; "
          f"reward={mean_reward_s:.4f}s gen={mean_gen_s:.2f}s) -> "
          f"{'PASS' if gate_timing else 'FAIL'}")
    print(f"[summarize] diagnostics: {json.dumps(diag, ensure_ascii=False)}")
    print(f"[summarize] summary -> {out_path}")
    print(f"[summarize] ALL {'PASS' if summary['all_pass'] else 'FAIL'} "
          f"(exit {0 if summary['all_pass'] else 1})")
    return 0 if summary["all_pass"] else 1


def main() -> int:
    args = parse_args()
    random.seed(args.seed)  # 仅影响非采样逻辑；样本抽样用独立 Random(seed)
    if args.stage == "dryrun":
        return run_dryrun(args)
    if args.stage == "timing":
        return run_timing(args)
    if args.stage == "gate":
        return run_gate(args)
    return run_summarize(args)


if __name__ == "__main__":
    sys.exit(main())
