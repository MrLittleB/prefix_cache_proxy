"""Router 测试：策略决策（同前缀、分布、会话 key、稳定性、过载）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from prefix_hash_router.router.context import RequestContext
from prefix_hash_router.router.backend import Backend
from prefix_hash_router.router import ChainedPolicy
from prefix_hash_router.router.policies import (
    SessionAffinityPolicy, PrefixHashPolicy, ConsistentHashPolicy,
    RoundRobinPolicy, OverloadGuard,
)


def _ctx(system=None, user="hi", session=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    headers = {}
    if session:
        headers["x-session-id"] = session
    return RequestContext(headers=headers, raw_body=b"", parsed_body={"messages": msgs}, session_key=None)


def test_same_prefix_same_rank():
    p = ChainedPolicy([SessionAffinityPolicy(8), PrefixHashPolicy(8)])
    b1 = p.route(_ctx(system="共享前缀" * 100))
    b2 = p.route(_ctx(system="共享前缀" * 100))
    assert b1.dp_rank == b2.dp_rank


def test_distinct_prefix_spread():
    p = ChainedPolicy([SessionAffinityPolicy(8), PrefixHashPolicy(8)])
    ranks = {p.route(_ctx(system=f"独立前缀-{i}" * 20)).dp_rank for i in range(64)}
    assert len(ranks) >= 4


def test_session_key_priority():
    p = ChainedPolicy([SessionAffinityPolicy(8), PrefixHashPolicy(8)])
    b = _ctx(system="共享前缀" * 50, session="sess-1")
    r1 = p.route(b).dp_rank
    r2 = p.route(_ctx(system="完全不同" * 50, session="sess-1")).dp_rank
    r3 = p.route(_ctx(system="共享前缀" * 50, session="sess-2")).dp_rank
    assert r1 == r2   # 同 session 同 rank（即使 body 不同）
    assert r1 != r3   # 不同 session 尽量不同


def test_key_stability_not_tail():
    """多轮：user 每轮变，但 system 固定 → key 应稳定落同一 rank。"""
    p = ChainedPolicy([SessionAffinityPolicy(8), PrefixHashPolicy(8)])
    sysp = "稳定系统前缀" * 80
    r1 = p.route(_ctx(system=sysp, user="第1轮")).dp_rank
    r2 = p.route(_ctx(system=sysp, user="第2轮")).dp_rank
    assert r1 == r2


def test_round_robin():
    p = ChainedPolicy([SessionAffinityPolicy(4), RoundRobinPolicy(4)])
    p.route(_ctx(system="x")), p.route(_ctx(system="x"))
    # round-robin 是兜底，无 session 时轮转（前三->0,1,2）
    a = p.route(_ctx(user="a")).dp_rank
    b = p.route(_ctx(user="b")).dp_rank
    c = p.route(_ctx(user="c")).dp_rank
    assert len({a, b, c}) >= 2


def test_overload_guard_keeps_affinity_when_not_hot():
    # 绝对阈值 min_load=5：首选 rank 负载 < 5 不干预，保持哈希命中
    load = [3, 3, 3, 3, 3, 3, 3, 3]
    p = ChainedPolicy([OverloadGuard(PrefixHashPolicy(8), load=load, load_skew=1.5, min_load=5)])
    b = p.route(_ctx(system="共享前缀" * 100))
    assert b.dp_rank == _prefix_key_of(_ctx(system="共享前缀" * 100), 8)

    # 即便某 rank 负载 4 但整体都 < min_load，也应守住哈希粘性（不被相对平均误判）
    load2 = [0, 0, 0, 0, 4, 0, 0, 0]
    p2 = ChainedPolicy([OverloadGuard(PrefixHashPolicy(8), load=load2, load_skew=1.5, min_load=5)])
    b2 = p2.route(_ctx(system="共享前缀" * 100))
    assert b2.dp_rank == _prefix_key_of(_ctx(system="共享前缀" * 100), 8)


def test_overload_guard_spills_when_hot():
    # 首选 rank(哈希到4)负载 100 >= min_load=5 → 真过载 → spill 到非 4
    load = [1, 1, 1, 1, 100, 1, 1, 1]
    from prefix_hash_router.router.policies._hash import hash_to_rank
    sys_text = next(t for i in range(3000) if (t := f"spill-{i}" * 20) and hash_to_rank(t, 8) == 4)
    p = ChainedPolicy([OverloadGuard(PrefixHashPolicy(8), load=load, load_skew=1.5, min_load=5)])
    b = p.route(_ctx(system=sys_text))
    assert b.dp_rank != 4


def test_session_key_anti_collapse_shared_prefix():
    """SGLang #31170 反塌缩：不同 session key 共享同一大前缀 → 分散到多个 rank。"""
    p = ChainedPolicy([SessionAffinityPolicy(8), PrefixHashPolicy(8)])
    shared = "完全相同共享前缀" * 200
    ranks = {p.route(_ctx(system=shared, session=f"session-{i}")).dp_rank for i in range(128)}
    assert len(ranks) >= 4


def _prefix_key_of(ctx, dp):
    from prefix_hash_router.router.policies._keys import extract_prefix_key
    from prefix_hash_router.router.policies._hash import hash_to_rank
    return hash_to_rank(extract_prefix_key(ctx), dp)


if __name__ == "__main__":
    test_same_prefix_same_rank()
    test_distinct_prefix_spread()
    test_session_key_priority()
    test_key_stability_not_tail()
    test_round_robin()
    test_overload_guard_keeps_affinity_when_not_hot()
    test_overload_guard_spills_when_hot()
    test_session_key_anti_collapse_shared_prefix()
    print("router 全部测试通过")
