#!/usr/bin/env python
"""RL Phase A rollout sampler（薄封装 LocateAnythingWorker）。

⚠️ 禁令：禁止把 generation_mode 改为 "fast"/"hybrid" 用于训练采样（off-policy）。
"slow"（NTP，逐 token 采样）是 Phase A/B 唯一合法训练分布。
采样四件套不许改：slow / top_p=1.0 / top_k=0 / repetition_penalty=1.0（temperature 可调）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 保证能 import 仓库根的 locateanything_worker（脚本可从任意 cwd 运行）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# RL 采样参数（设计定值；除 temperature 外不许改）
RL_GENERATION_KW = {
    "generation_mode": "slow",  # NTP；on-policy 硬约束
    "top_p": 1.0,
    "top_k": 0,
    "repetition_penalty": 1.0,
}


def make_worker(model_path: str, device: str = "cuda"):
    from locateanything_worker import LocateAnythingWorker  # 延迟导入（重量级）

    return LocateAnythingWorker(model_path, device=device)


def sample_tile(worker, image, prompt: str, n: int, *, temperature: float = 1.0,
                max_new_tokens: int = 512) -> list:
    """对一个 tile 独立采样 n 次。

    max_new_tokens=512 ≈ 20 框预算；超长截断由 reward 端 n_raw>20 判非法承接。
    slow 模式无批量路径，逐条采样并计时。
    返回 [{answer, gen_seconds, sequences}]。
    """
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        res = worker.predict(
            image=image,
            question=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=False,
            return_ids=True,
            **RL_GENERATION_KW,
        )
        dt = time.perf_counter() - t0
        if not isinstance(res, dict) or "sequences" not in res:
            raise RuntimeError(
                "generate() did not return sequences; checkpoint generate() missing return_ids"
            )
        out.append({"answer": res["answer"], "gen_seconds": dt, "sequences": res["sequences"]})
    return out
