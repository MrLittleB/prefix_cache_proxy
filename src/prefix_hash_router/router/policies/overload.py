"""OverloadGuard：过载保护，参考 SGLang #31170 的 overload guard。

当首选 rank 负载 > 平均×load_skew 时，按 HRW 候选优先级依次找落在负载阈值内的
下一个 rank（spill），避免"共享大前缀"把所有请求压到单 rank 的热点（D3/D7）。

用法：
  guard = OverloadGuard(PrefixHashPolicy(8), load, load_skew=1.5, atom_key=<session/prefix key>)
atom_key 用于按 HRW 重新排序候选；若无 atom_key，退化为"就近找最闲"。
"""
from __future__ import annotations
import sys

from ..context import RequestContext
from ..backend import Backend
from ._hash import blake2b_int

# 过载 spill 诊断开关（由 main.py 在 --debug-rank 时开启）。
# 用于定位"为什么 rank 在乱跳"——往往是被 OverloadGuard 误判 spill。
_OVERLOAD_DEBUG = False


def set_overload_debug(on: bool = True) -> None:
    """开启/关闭过载 spill 的诊断输出。"""
    global _OVERLOAD_DEBUG
    _OVERLOAD_DEBUG = on


class OverloadGuard:
    def __init__(self, inner, load=None, load_provider=None, load_skew: float = 1.5,
                 min_load: float = 5.0, atom_from_ctx=None):
        """inner: 被包装的 Policy；atom_from_ctx: ctx -> 原子键(可选，用于 HRW spill)。

        load 和 load_provider 二选一：
          - load:          静态负载列表（测试用）
          - load_provider:  可调用 -> 当前负载列表（生产用，从 MetricsCollector 实时取，
                            running/waiting 加权后喂进来）
        min_load:  过载的"绝对门槛"（running+waiting）。首选 rank 负载 < min_load 时，
                  视为低负载、能扛住，一律不干预（保护 radix 前缀一致性，避免"1 vs 0"误判）。
                  只有负载 >= min_load 之后，才进一步用"相对平均阈值(avg*skew)"判定是否真过载。
                  （默认 5，且不等价于"到 5 就 spill"——还要满足高于平均才算真过载。）
        """
        self._inner = inner
        self._static_load = load
        self._provider = load_provider
        self._skew = load_skew
        self._min_load = min_load
        self._atom_from_ctx = atom_from_ctx

    def _current_load(self):
        if self._provider is not None:
            # provider 返回 None 表示"无负载数据"→ 跳过过载保护(退化为纯路由)
            return self._provider() or []
        return self._static_load or []

    def route(self, ctx: RequestContext) -> Backend | None:
        b = self._inner.route(ctx)
        if b is None:
            return None
        load = self._current_load()
        n = len(load)
        if n == 0:
            return b
        pref = b.dp_rank % n
        # 第一层：绝对门槛。首选 rank 负载 < min_load 时，视为"低负载、能扛住"，
        # 一律不干预（保护 radix 前缀一致性，避免"1 vs 0"误判）。默认 min_load=5。
        if load[pref] < self._min_load:
            return Backend(dp_rank=pref)

        # 第二层：过了绝对门槛后，才用"相对平均阈值"判定是否真过载。
        # 即在负载 >= min_load 的基础上，再要求明显高于其它 DP 平均（avg*skew）才 spill。
        avg = sum(load) / n
        threshold = avg * self._skew
        if load[pref] <= threshold:
            return Backend(dp_rank=pref)

        # 真过载（>= min_load 且 > avg*skew）→ spill
        if _OVERLOAD_DEBUG:
            sys.stderr.write(
                f"[overload] SPILL 触发: 首选rank={pref} 负载={load[pref]:.2f} "
                f"绝对门槛={self._min_load} 相对阈值={threshold:.2f} (avg={avg:.2f} skew={self._skew}) "
                f"全负载={[round(x,2) for x in load]} "
                f"inner_type={type(self._inner).__name__}\n"
            )
        atom_key = self._atom_from_ctx(ctx) if self._atom_from_ctx else None
        if atom_key is not None:
            new_rank = self._spill_by_hrw(load, pref, n, threshold, atom_key)
            if _OVERLOAD_DEBUG:
                sys.stderr.write(f"    -> spill 到 rank={new_rank} (via HRW, atom_key={atom_key!r})\n")
            return Backend(dp_rank=new_rank)
        new_rank = self._spill_by_load(load, pref, n, threshold)
        if _OVERLOAD_DEBUG:
            sys.stderr.write(f"    -> spill 到 rank={new_rank} (via load)\n")
        return Backend(dp_rank=new_rank)

    def _spill_by_hrw(self, load, pref, n, threshold, atom_key):
        # 按 HRW 对所有 rank 评分排序（越靠前越亲和），跳到负载低于相对阈值(未过载)的候选
        candidates = sorted(
            range(n),
            key=lambda r: (-blake2b_int(f"{atom_key}#rank{r}", digest_size=8), r),
        )
        for r in candidates:
            if load[r] < threshold:
                return r
        return min(candidates, key=lambda r: load[r])  # 全过载则选最闲

    def _spill_by_load(self, load, pref, n, threshold):
        best = pref
        best_load = load[pref]
        for k in range(1, n):
            idx = (pref + k) % n
            if load[idx] < best_load:
                best_load = load[idx]
                best = idx
        return best
