#!/usr/bin/env python
"""RL Phase A reward for LocateAnything-SAT HBB detection (design v2.1).

纯 CPU、无模型依赖；HBB only（OBB 旋转 IoU 属 Phase C2，不在本模块）。

模型输出 token 格式（与 locateanything_worker / 训练 JSONL 一致）：
  框:     <ref>label</ref><box><x1><y1><x2><y2></box>   坐标为 [0,1000] 归一化整数
  无目标: <box>none</box>

奖励分支（v2.1，顺序执行，命中即返回）：
  1. parse invalid            → r_main = 0.0
  2. len(gt)==0（空 tile）     → none: 1.0；否则 max(0, 0.3 − 0.1·n_raw)
  3. is_none 且 len(gt)>0     → r_main = none_penalty = −0.5
  4. n_raw==0（合法零框）      → r_main = 0.0
  5. 常规: NMS(按生成顺序贪心, 同 label, IoU>thr 抑制) → kept
     r_main = F1_soft(label-aware) × count_ratio × dup_term
       P_soft = mean_i max_{j: label_j==label_i} IoU(pred_i, gt_j)
       R_soft = mean_j max_{i: label_i==label_j} IoU(pred_i, gt_j)
       count_ratio = min(1, (N_gt+1)/(N_pred+1))   （NMS 后框数）
       dup_term    = 1 − lam_dup·(n_raw − N_pred)/n_raw
     对照列 r_eff：同公式但 IoU 换 IoU_eff = max(IoU, 0.5·center_sim)，
       center_sim = exp(−d²/(2·(0.25·sqrt(area_gt))²))；仅诊断输出，不进任何门。

f1_at_05 与 eval_baseline_dota.py::match_boxes 同构（同 label、IoU≥0.5、贪心一对一），
是奖励门的评测侧锚点。
"""
from __future__ import annotations

import math
import re

import torch
from torchvision.ops import box_iou

MAX_BOXES = 20  # 框数上限 = 格式失败阈值

_RE_PAIR = re.compile(r"<ref>(.*?)</ref>\s*<box>(.*?)</box>", re.S)
_RE_BOX = re.compile(r"<box>(.*?)</box>", re.S)
_RE_REF = re.compile(r"<ref>(.*?)</ref>", re.S)
_RE_COORD = re.compile(r"<(-?\d+)>")


def parse_answer(text: str, tile_size: int = 448) -> dict:
    """解析模型/GT 答案文本 → {valid, is_none, boxes, n_raw}。

    boxes: list[(label, x1, y1, x2, y2)]，坐标已 token→像素（c·tile_size/1000，clamp 到
    [0, tile_size]）；退化框（x2<=x1 或 y2<=y1）从 boxes 丢弃但仍计入 n_raw。
    n_raw: 解析出的框总数（丢弃前）。valid 且 n_raw>20 → valid=False。
    """
    blocks = [m.group(1).strip() for m in _RE_BOX.finditer(text)]
    refs = _RE_REF.findall(text)
    pairs = _RE_PAIR.findall(text)

    # none 答案：全部 box 块均为 "none" 且无 ref（混排 none+实体框判非法，防博弈）
    if blocks and not refs and all(b == "none" for b in blocks):
        return {"valid": True, "is_none": True, "boxes": [], "n_raw": 0}

    invalid = {"valid": False, "is_none": False, "boxes": [], "n_raw": len(pairs)}
    if not blocks or not pairs:
        return invalid  # 无任何 box 块
    if len(refs) != len(pairs) or len(blocks) != len(pairs):
        return invalid  # 有 ref 无 box / 有 box 无 ref / none 混排

    boxes = []
    for label, coord_text in pairs:
        coords = [int(c) for c in _RE_COORD.findall(coord_text)]
        if len(coords) != 4:
            return invalid
        x1, y1, x2, y2 = (
            min(max(c * tile_size / 1000.0, 0.0), float(tile_size)) for c in coords
        )
        if x2 > x1 and y2 > y1:
            boxes.append((label, x1, y1, x2, y2))

    n_raw = len(pairs)
    valid = n_raw <= MAX_BOXES
    return {
        "valid": valid,
        "is_none": False,
        "boxes": boxes if valid else [],
        "n_raw": n_raw,
    }


def nms_order(boxes: list, iou_thr: float = 0.7) -> list:
    """无分数 NMS：按生成顺序贪心。返回保留索引（原顺序）。

    后发框与任一已保留的同 label 框 IoU > iou_thr 即被抑制。
    """
    if not boxes:
        return []
    coords = torch.tensor([[b[1], b[2], b[3], b[4]] for b in boxes], dtype=torch.float64)
    iou_mat = box_iou(coords, coords)
    keep = []
    for i in range(len(boxes)):
        suppressed = False
        for j in keep:
            if boxes[j][0] == boxes[i][0] and iou_mat[i, j].item() > iou_thr:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return keep


def _label_masked_iou(pred_boxes: list, gt_boxes: list) -> torch.Tensor:
    """[Np, Ng] IoU 矩阵；异 label 对置 0（label-aware：IoU 只在同 label 对上有效）。"""
    p = torch.tensor([[b[1], b[2], b[3], b[4]] for b in pred_boxes], dtype=torch.float64)
    g = torch.tensor([[b[1], b[2], b[3], b[4]] for b in gt_boxes], dtype=torch.float64)
    iou_mat = box_iou(p, g)
    mask = torch.tensor(
        [[pb[0] == gb[0] for gb in gt_boxes] for pb in pred_boxes], dtype=torch.bool
    )
    return iou_mat * mask


def _center_sim(pred_boxes: list, gt_boxes: list) -> torch.Tensor:
    """[Np, Ng] 中心距高斯相似度，尺度锚 GT 面积开方：exp(−d²/(2·σ²)), σ=0.25·sqrt(area_gt)。"""
    p = torch.tensor(
        [[(b[1] + b[3]) / 2.0, (b[2] + b[4]) / 2.0] for b in pred_boxes],
        dtype=torch.float64,
    )
    centers, sigmas = [], []
    for b in gt_boxes:
        centers.append([(b[1] + b[3]) / 2.0, (b[2] + b[4]) / 2.0])
        sigmas.append(0.25 * math.sqrt((b[3] - b[1]) * (b[4] - b[2])))
    g = torch.tensor(centers, dtype=torch.float64)
    sigma = torch.tensor(sigmas, dtype=torch.float64)
    d2 = ((p[:, None, :] - g[None, :, :]) ** 2).sum(dim=-1)
    return torch.exp(-d2 / (2.0 * sigma[None, :] ** 2))


def _result(r_main, r_eff, f1_soft, count_ratio, dup_term, n_raw, n_nms,
            is_none, parse_ok, mean_max_iou) -> dict:
    return {
        "r_main": r_main,
        "r_eff": r_eff,
        "f1_soft": f1_soft,
        "count_ratio": count_ratio,
        "dup_term": dup_term,
        "n_raw": n_raw,
        "n_nms": n_nms,
        "is_none": is_none,
        "parse_ok": parse_ok,
        "mean_max_iou": mean_max_iou,
    }


def _reward_from_boxes(pred_boxes: list, gt_boxes: list, n_raw: int, *,
                       nms_thr: float = 0.7, lam_dup: float = 0.5) -> dict:
    """常规分支：已解析（非 none、n_raw>0）的框列表直接进奖励。selftest 用它做精确构造。"""
    kept_idx = nms_order(pred_boxes, nms_thr)
    kept = [pred_boxes[i] for i in kept_idx]
    n_nms = len(kept)
    if not kept:
        return _result(0.0, 0.0, 0.0, 0.0, 0.0, n_raw, n_nms, False, True, 0.0)

    iou_m = _label_masked_iou(kept, gt_boxes)  # [Np, Ng]
    p_soft = iou_m.max(dim=1).values.mean().item()
    r_soft = iou_m.max(dim=0).values.mean().item()
    f1_soft = 2.0 * p_soft * r_soft / (p_soft + r_soft) if (p_soft + r_soft) > 0 else 0.0

    n_gt = len(gt_boxes)
    count_ratio = min(1.0, (n_gt + 1) / (n_nms + 1))
    dup_term = 1.0 - lam_dup * (n_raw - n_nms) / n_raw
    r_main = f1_soft * count_ratio * dup_term

    # r_eff 对照列：IoU → IoU_eff（同 label mask 同样施加于 center_sim）
    mask = torch.tensor(
        [[pb[0] == gb[0] for gb in gt_boxes] for pb in kept], dtype=torch.bool
    )
    eff_m = torch.maximum(iou_m, 0.5 * _center_sim(kept, gt_boxes) * mask)
    pe = eff_m.max(dim=1).values.mean().item()
    re_ = eff_m.max(dim=0).values.mean().item()
    f1_eff = 2.0 * pe * re_ / (pe + re_) if (pe + re_) > 0 else 0.0
    r_eff = f1_eff * count_ratio * dup_term

    assert math.isfinite(r_main) and math.isfinite(r_eff), f"reward not finite: {r_main} {r_eff}"
    assert -0.5 <= r_main <= 1.0, f"r_main out of range: {r_main}"
    return _result(r_main, r_eff, f1_soft, count_ratio, dup_term, n_raw, n_nms,
                   False, True, p_soft)


def compute_reward(text: str, gt: list, *, nms_thr: float = 0.7, lam_dup: float = 0.5,
                   none_penalty: float = -0.5, tile_size: int = 448) -> dict:
    """主入口。gt: list[(label, x1, y1, x2, y2)]（像素坐标，与 parse_answer 同空间）。"""
    gt = [g for g in gt if g[3] > g[1] and g[4] > g[2]]  # 防御：丢弃不可匹配的退化 GT
    parse = parse_answer(text, tile_size)

    if not parse["valid"]:
        return _result(0.0, 0.0, 0.0, 0.0, 0.0, parse["n_raw"], 0,
                       False, False, 0.0)

    n_raw = parse["n_raw"]
    is_none = parse["is_none"]

    if len(gt) == 0:  # 空 tile：none 得满分；吐框按数量衰减；不进 soft-F1
        r = 1.0 if is_none else max(0.0, 0.3 - 0.1 * n_raw)
        return _result(r, r, 0.0, 0.0, 0.0, n_raw, 0, is_none, True, 0.0)

    if is_none:  # 有目标却答 none
        return _result(none_penalty, none_penalty, 0.0, 0.0, 0.0, n_raw, 0,
                       True, True, 0.0)

    if n_raw == 0:  # 合法零框且非 none（防御路径，当前 parser 下不可达）
        return _result(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, False, True, 0.0)

    return _reward_from_boxes(parse["boxes"], gt, n_raw, nms_thr=nms_thr, lam_dup=lam_dup)


def f1_at_05(pred: list, gt: list) -> float:
    """IoU≥0.5、同 label、贪心一对一匹配（按 IoU 降序，每 pred/gt 至多一次）的 F1。

    语义对齐 eval_baseline_dota.py::match_boxes；逐样本值，无 per-class 聚合。
    """
    pred = [p for p in pred if p[3] > p[1] and p[4] > p[2]]
    gt = [g for g in gt if g[3] > g[1] and g[4] > g[2]]
    if not pred or not gt:
        return 0.0

    scored = []
    for i, (pl, pb) in enumerate(pred):
        for j, (gl, gb) in enumerate(gt):
            if pl != gl:
                continue
            ix1, iy1 = max(pb[1], gb[1]), max(pb[2], gb[2])
            ix2, iy2 = min(pb[3], gb[3]), min(pb[4], gb[4])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            union = (pb[3] - pb[1]) * (pb[4] - pb[2]) + (gb[3] - gb[1]) * (gb[4] - gb[2]) - inter
            if union <= 0:
                continue
            sc = inter / union
            if sc >= 0.5:
                scored.append((sc, i, j))
    scored.sort(reverse=True)

    matched_pred, matched_gt = set(), set()
    for sc, i, j in scored:
        if i not in matched_pred and j not in matched_gt:
            matched_pred.add(i)
            matched_gt.add(j)
    tp = len(matched_pred)
    p = tp / len(pred)
    r = tp / len(gt)
    return 2.0 * p * r / (p + r) if (p + r) > 0 else 0.0


# --------------------------------------------------------------------------- #
# selftest：python scripts/rl/reward.py --selftest（无外部数据依赖）
# --------------------------------------------------------------------------- #

def _approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def _ans(label: str, x1: int, y1: int, x2: int, y2: int) -> str:
    return f"<ref>{label}</ref><box><{x1}><{y1}><{x2}><{y2}></box>"


def selftest() -> None:
    # token 空间用例统一 tile_size=1000 → token 数值 == 像素数值，消除量化舍入
    T = 1000
    gt1 = [("a", 10.0, 10.0, 50.0, 50.0)]

    # ---- parse_answer 单元 ----
    pa = parse_answer(_ans("a", 100, 200, 145, 225), tile_size=448)
    assert pa["valid"] and not pa["is_none"] and pa["n_raw"] == 1
    (lbl, x1, y1, x2, y2) = pa["boxes"][0]
    assert lbl == "a" and _approx(x1, 44.8) and _approx(y1, 89.6) \
        and _approx(x2, 64.96) and _approx(y2, 100.8), pa["boxes"]

    pa = parse_answer(_ans("a", 1200, -5, 500, 500), tile_size=448)  # clamp
    assert pa["boxes"][0][1] == 448.0 and pa["boxes"][0][2] == 0.0

    pa = parse_answer(_ans("a", 500, 500, 100, 100), tile_size=T)  # 退化框：丢弃但计 n_raw
    assert pa["valid"] and pa["boxes"] == [] and pa["n_raw"] == 1

    assert parse_answer("<box>none</box>", T)["is_none"] is True
    assert parse_answer("<box> none </box>", T)["is_none"] is True
    assert parse_answer("locate nothing.", T)["valid"] is False  # 无 box 块
    assert parse_answer("<ref>a</ref>", T)["valid"] is False  # 有 ref 无 box
    assert parse_answer("<ref>a</ref> blah <box><1><2><3><4></box>", T)["valid"] is False
    assert parse_answer("<ref>a</ref><box>none</box>", T)["valid"] is False  # ref+none 混排
    assert parse_answer(_ans("a", 1, 1, 2, 2) + "<box>none</box>", T)["valid"] is False
    assert parse_answer("<ref>a</ref><box><1><2><3></box>", T)["valid"] is False  # 坐标数≠4
    assert parse_answer("".join(_ans("a", i, i, i + 5, i + 5) for i in range(20)), T)["valid"]
    assert parse_answer("".join(_ans("a", i, i, i + 5, i + 5) for i in range(21)), T)["valid"] is False

    # ---- nms_order 单元 ----
    assert nms_order([("a", 0, 0, 10, 10), ("a", 0, 0, 10, 10), ("b", 0, 0, 10, 10)]) == [0, 2]
    assert nms_order([("a", 0, 0, 10, 10), ("a", 20, 20, 30, 30)]) == [0, 1]

    # ---- compute_reward 分支（计划表用例，精确期望）----
    r = compute_reward(_ans("a", 10, 10, 50, 50), gt1, tile_size=T)
    assert _approx(r["r_main"], 1.0) and r["parse_ok"] and not r["is_none"], r

    r = compute_reward("<box>none</box>", [], tile_size=T)
    assert _approx(r["r_main"], 1.0) and r["is_none"], r

    r = compute_reward(_ans("a", 10, 10, 50, 50), [], tile_size=T)
    assert _approx(r["r_main"], 0.2), r

    r = compute_reward("".join(_ans("a", 10 * i, 10 * i, 10 * i + 9, 10 * i + 9)
                               for i in range(5)), [], tile_size=T)
    assert _approx(r["r_main"], 0.0), r

    r = compute_reward("<box>none</box>", gt1, tile_size=T)
    assert _approx(r["r_main"], -0.5), r

    r = compute_reward(_ans("a", 10, 10, 50, 50) + _ans("a", 10, 10, 50, 50), gt1, tile_size=T)
    assert _approx(r["r_main"], 0.75) and r["n_nms"] == 1, r  # 重复框：dup_term=0.75

    r = compute_reward(_ans("b", 10, 10, 50, 50), gt1, tile_size=T)
    assert _approx(r["r_main"], 0.0), r  # 错类：同 label IoU 全 0

    r = compute_reward("完全乱掉的输出 no boxes here", gt1, tile_size=T)
    assert _approx(r["r_main"], 0.0) and not r["parse_ok"], r

    r = compute_reward("".join(_ans("a", i, i, i + 5, i + 5) for i in range(21)),
                       gt1, tile_size=T)
    assert _approx(r["r_main"], 0.0) and not r["parse_ok"], r  # >20 框 → 非法

    # ---- 过量吐框（计划"多吐框"用例的两个可构造替身；见模块注释）----
    # 几何事实：与同一 GT 的 IoU 均 ≥0.9 的两个框，彼此 IoU ≥ ~0.8 > NMS 阈值 0.7，
    # 故"10 个各 IoU=0.9 且两两 IoU<0.7 的框"不可构造（计划表 ≈0.172 的原始设定）。
    # 替身 1（NMS 坍缩）：10 个各 IoU=0.9 的框 → kept=1，
    #   r = F1(0.9) × count(min(1,2/2)=1) × dup(1−0.5·9/10=0.55) = 0.495
    gt_big = [("a", 0.0, 0.0, 100.0, 100.0)]
    preds = [("a", float(k), 0.0, float(90 + k), 100.0) for k in range(10)]
    r = _reward_from_boxes(preds, gt_big, 10)
    assert r["n_nms"] == 1 and _approx(r["r_main"], 0.495), r
    # 替身 2（全保留 + 计数惩罚）：10 个各 IoU=0.55、两两 IoU<0.7（5 竖条 + 5 横条），
    #   r = F1(0.55) × count(2/11) × dup(1) = 0.1
    strips = [("a", 11.25 * k, 0.0, 11.25 * k + 55, 100.0) for k in range(5)]
    strips += [("a", 0.0, 11.25 * k, 100.0, 11.25 * k + 55) for k in range(5)]
    r = _reward_from_boxes(strips, gt_big, 10)
    assert r["n_nms"] == 10 and _approx(r["r_main"], 0.1), r

    # ---- f1_at_05 ----
    assert _approx(f1_at_05([("a", 0, 0, 10, 10)], [("a", 0, 0, 10, 10)]), 1.0)
    # 1 GT + 2 个 IoU=0.6 同 label 框 → TP=1, P=0.5, R=1 → F1≈0.667
    f = f1_at_05([("a", 0, 0, 6, 10), ("a", 4, 0, 10, 10)], [("a", 0, 0, 10, 10)])
    assert _approx(f, 2.0 * 0.5 * 1.0 / 1.5), f
    assert _approx(f1_at_05([("b", 0, 0, 10, 10)], [("a", 0, 0, 10, 10)]), 0.0)
    assert _approx(f1_at_05([], [("a", 0, 0, 10, 10)]), 0.0)
    assert _approx(f1_at_05([("a", 0, 0, 10, 10)], []), 0.0)

    print("reward.py selftest: all cases passed")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv[1:]:
        selftest()
    else:
        print(__doc__)
