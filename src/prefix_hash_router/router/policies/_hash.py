"""共享：非随机稳定哈希（blake2b）与一致性哈希环。"""
from __future__ import annotations

import bisect
import hashlib


def blake2b_int(key: str, digest_size: int = 8) -> int:
    if not key:
        key = "\x00"
    return int.from_bytes(
        hashlib.blake2b(key.encode("utf-8"), digest_size=digest_size).digest(), "big",
    )


def hash_to_rank(key: str, dp_size: int) -> int:
    if dp_size <= 0:
        raise ValueError("dp_size must be >= 1")
    return blake2b_int(key) % dp_size


def consistent_hash_ring(key: str, dp_size: int, replicas: int = 100) -> int:
    if dp_size <= 0:
        raise ValueError("dp_size must be >= 1")
    ring: list[tuple[int, int]] = []
    for rank in range(dp_size):
        for i in range(replicas):
            ring.append((blake2b_int(f"{rank}#{i}"), rank))
    ring.sort()
    points = [p for p, _ in ring]
    target = blake2b_int(key)
    idx = bisect.bisect_left(points, target)
    if idx == len(points):
        idx = 0
    return ring[idx][1]


def hrw_hash_rank(key: str, dp_size: int) -> int:
    """Rendezvous / Highest Random Weight (HRW) hashing。

    参考 SGLang #31170：用**稳定 rank id** 对每个候选 rank 独立评分，
    选评分最高者。rank 下线/增减时，只重映射受影响 key，而非 hash%N 的全体重排。
    """
    if dp_size <= 0:
        raise ValueError("dp_size must be >= 1")
    best_rank = 0
    best_score = -1
    for rank in range(dp_size):
        score = blake2b_int(f"{key}#rank{rank}", digest_size=8)
        if score > best_score:
            best_score = score
            best_rank = rank
    return best_rank
