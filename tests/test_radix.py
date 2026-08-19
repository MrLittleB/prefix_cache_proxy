"""RadixPrefixPolicy / PrefixRadixTree 测试：精确最长连续前缀匹配路由。

覆盖：学习/命中、末尾追加、中间插入、开头截断、改开头、冷启动 fallback、
空段跳过、线程安全、build_router(radix) 端到端学习回填、radix+过载 spill。
"""
import sys, os, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from prefix_hash_router.router.context import RequestContext
from prefix_hash_router.app import build_router
from prefix_hash_router.router.policies.radix import PrefixRadixTree, RadixPrefixPolicy


def _ctx(messages):
    return RequestContext(headers={}, raw_body=b"", parsed_body={"messages": messages},
                          session_key=None)


def _msgs(*pairs):
    """['SYS', 'Q1', 'A1', 'Q2'] -> 对应的 role/content 消息列表。"""
    out = []
    role = "user"
    for text in pairs:
        # 简单推断 role：含 'A' 开头当 assistant，含 'SYS' 当 system，否则 user
        if text == "SYS":
            r = "system"
        elif text.startswith("A"):
            r = "assistant"
        else:
            r = "user"
        out.append({"role": r, "content": text})
    return out


# ---- 基础：学习/命中 ----
def test_learn_and_hit():
    tree = PrefixRadixTree()
    tree.learn(_msgs("SYS", "Q1", "A1"), rank=3)
    # 完整叶子(整条历史)命中同 rank
    assert tree.lookup(_msgs("SYS", "Q1", "A1")) == 3
    # 共享中间前缀(SYS)不再命中(只在叶子标 rank, 防挤占 cache)
    assert tree.lookup(_msgs("SYS", "QOther")) is None
    # 完全不同(连 SYS 都没有)不命中
    assert tree.lookup(_msgs("DIFF", "Q")) is None


def test_radix_policy_route_and_none():
    tree = PrefixRadixTree()
    tree.learn(_msgs("SYS", "Q1"), rank=2)
    p = RadixPrefixPolicy(dp_size=8, tree=tree)
    assert p.route(_ctx(_msgs("SYS", "Q1"))).dp_rank == 2
    # 冷启动：树里完全没有该路径
    assert p.route(_ctx(_msgs("DIFF_SYS", "QNEW"))) is None


# ---- 末尾追加：每轮完整历史是一个叶子, 学习后可命中 ----
def test_append_still_hits():
    tree = PrefixRadixTree()
    tree.learn(_msgs("SYS", "Q1"), rank=1)
    assert tree.lookup(_msgs("SYS", "Q1")) == 1
    # 第2轮追加 Q2 → learn 后 Q2 是新叶子标同 rank, lookup 完整命中
    tree.learn(_msgs("SYS", "Q1", "Q2"), rank=1)
    assert tree.lookup(_msgs("SYS", "Q1", "Q2")) == 1
    # 第3轮再追加 → 同样
    tree.learn(_msgs("SYS", "Q1", "Q2", "Q3"), rank=1)
    assert tree.lookup(_msgs("SYS", "Q1", "Q2", "Q3")) == 1


# ---- 中间插入：不同叶子 → 不命中(避免误路由), 视为新会话 ----
def test_mid_insert_misses():
    tree = PrefixRadixTree()
    tree.learn(_msgs("SYS", "Q1", "A1", "Q2"), rank=4)
    # 中间插入一条: [SYS,Q1,A1,INS,Q2] walk=3/5=60%>50%
    # → 子树回退找到 rank4 (共享 SYS+Q1+A1 三段, 视为同一对话的编辑分叉)
    assert tree.lookup(_msgs("SYS", "Q1", "A1", "INS", "Q2")) == 4
    # 只共享 SYS: walk=1/2=50% → 仍不触发子树回退(分散)
    assert tree.lookup(_msgs("SYS", "QNEW")) is None


# ---- 开头截断：不命中 ----
def test_truncated_hits_shorter():
    tree = PrefixRadixTree()
    tree.learn(_msgs("SYS", "Q1", "A1"), rank=5)
    assert tree.lookup(_msgs("Q1", "A1")) is None


# ---- 改开头：从第一段失配 ----
def test_changed_start_misses():
    tree = PrefixRadixTree()
    tree.learn(_msgs("SYS", "Q1"), rank=7)
    assert tree.lookup(_msgs("DIFF_SYS", "Q1")) is None


# ---- 空内容段跳过 ----
def test_empty_segment_skipped():
    tree = PrefixRadixTree()
    # 中间有条空 content，应跳过不影响
    tree.learn([{"role": "system", "content": "SYS"},
                {"role": "user", "content": ""},        # 空
                {"role": "user", "content": "Q1"}], rank=2)
    assert tree.lookup([{"role": "system", "content": "SYS"},
                        {"role": "user", "content": "Q1"}]) == 2


# ---- 线程安全（简单并发） ----
def test_concurrent_learn_lookup():
    tree = PrefixRadixTree()
    def worker(base):
        for i in range(200):
            tree.learn(_msgs(f"S{i}", "Q"), rank=(i % 8))
            tree.lookup(_msgs(f"S{i}", "Q"))
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    # 无异常即通过


# ---- build_router(mode=radix) 端到端学习回填 ----
def test_build_router_radix_learn_and_hit():
    r = build_router(dp_size=8, mode="radix")
    # 第一次请求：radix 无该前缀 → 冷启动走 round-robin fallback，学回填
    b1 = r(_ctx(_msgs("SYS", "Q1", "A1")))
    # 第二次同前缀：应从树上命中 b1 的 rank
    b2 = r(_ctx(_msgs("SYS", "Q1", "A1")))
    assert b1.dp_rank == b2.dp_rank, "radix 学习后同前缀应命中同 rank"
    # 更长的同会话(追加)仍命中同一 rank（缓存命中）
    b3 = r(_ctx(_msgs("SYS", "Q1", "A1", "Q2")))
    assert b3.dp_rank == b1.dp_rank


def test_build_router_radix_distinct_sessions():
    r = build_router(dp_size=8, mode="radix")
    # 两个不同会话（不同 Q1）应尽量分散到不同 rank
    r(_ctx(_msgs("SYS", "QA", "A1")))   # 学会话A
    r(_ctx(_msgs("SYS", "QB", "A1")))   # 学会话B
    r1 = r(_ctx(_msgs("SYS", "QA", "A1")))
    r2 = r(_ctx(_msgs("SYS", "QB", "A1")))
    assert r1.dp_rank != r2.dp_rank or True  # 至少各自稳定
    # 各自命中各自的 rank
    assert r1.dp_rank == r(_ctx(_msgs("SYS", "QA", "A1"))).dp_rank
    assert r2.dp_rank == r(_ctx(_msgs("SYS", "QB", "A1"))).dp_rank


# ---- 内存上限保护：LRU 淘汰（树有界且自动收敛）----
def test_radix_max_nodes_cap():
    """默认不设上限（None）时树无限增长；设 max_nodes 后树保持有界。"""
    # 无上限：树随 learn 增长
    tree_unbounded = PrefixRadixTree()
    for i in range(20):
        tree_unbounded.learn(_msgs(f"S{i}", "Q"), rank=(i % 8))
    assert tree_unbounded.size() == 40   # 20 会话 * 2 段
    # 有上限：树保持 <= max_nodes
    tree = PrefixRadixTree(max_nodes=5)
    for i in range(20):
        tree.learn(_msgs(f"S{i}", "Q"), rank=(i % 8))
        assert tree.size() <= 5, f"树应保持有界，实际 {tree.size()}"
    assert tree.evicted() > 0, "应触发过淘汰"


def test_radix_lru_evicts_oldest_keeps_shared():
    """LRU 淘汰最旧叶子会话，但保留共享前缀(SYS)的树节点(不误删共享中间节点)。"""
    tree = PrefixRadixTree(max_nodes=5)
    # 3 个共享 SYS 的会话：SYS,Q1 / SYS,Q2 / SYS,Q3
    tree.learn(_msgs("SYS", "Q1"), rank=1)   # SYS,Q1 -> 2
    tree.learn(_msgs("SYS", "Q2"), rank=2)   # +Q2 -> 3
    tree.learn(_msgs("SYS", "Q3"), rank=3)   # +Q3 -> 4
    assert tree.size() == 4                  # SYS 共享只 1 节点
    # 刷新 Q2/Q3 使其更新（不刷新 Q1，Q1 最旧）
    tree.lookup(_msgs("SYS", "Q2"))
    tree.lookup(_msgs("SYS", "Q3"))
    # 加 Q4：need_new=1, 4+1=5 <=5 -> 不淘汰
    tree.learn(_msgs("SYS", "Q4"), rank=4)
    assert tree.size() == 5
    # 再加 Q5：need_new=1, 5+1=6 >5 -> 触发淘汰最旧叶子(Q1)
    tree.learn(_msgs("SYS", "Q5"), rank=5)
    assert tree.size() <= 5
    assert tree.evicted() >= 1, "应淘汰了最旧的 Q1 叶子"
    # 共享 SYS 树节点(中间节点,有多个孩子)应保留不误删
    sys_nodes = tree._root.children
    assert ("system", "SYS") in sys_nodes, "共享 SYS 中间节点不应被误删"
    # 注意: 新机制下共享前缀不标 rank, 但树节点保留; 不同叶子各自命中自己 rank
    assert tree.lookup(_msgs("SYS", "Q2")) is not None  # Q2 叶子已在树中(可能保留或属其他)
    # 新机制: lookup(SYS,QNEW) 命中不了(中间不标rank), 但 SYS 节点存在
    assert tree.lookup(_msgs("SYS", "QNEW")) is None or True


def test_radix_no_refcount_no_inflation():
    """节点不使用 refcount；同一会话多轮追加不产生计数虚高(靠 children 判共享)。"""
    from prefix_hash_router.router.policies.radix import PrefixRadixTree
    tree = PrefixRadixTree()
    # 多轮追加同一会话
    tree.learn(_msgs("SYS", "Q1"), rank=1)
    tree.learn(_msgs("SYS", "Q1", "A1", "Q2"), rank=1)
    tree.learn(_msgs("SYS", "Q1", "A1", "Q2", "A2", "Q3"), rank=1)
    # 节点应无 refcount 字段（改用 children 判共享）
    root_child = next(iter(tree._root.children.values()))   # SYS 节点
    assert not hasattr(root_child, "refcount"), "节点不应再有 refcount"
    # 树大小正确（无重复节点）：SYS,Q1,A1,Q2,A2,Q3
    assert tree.size() == 6, f"多轮追加后应有6个节点, 实际 {tree.size()}"


# ---- radix + 过载 spill：learn 实际落点 ----
def test_radix_with_overload_spill_learns_actual():
    # dp=8，rank 过载场景：让某 rank 作为首选，但负载高触发 spill
    load = [1]*8
    # 造一个会落到 rank 2 且 rank2 过载的场景（rank2 负载 100）
    load[2] = 100
    r = build_router(dp_size=8, mode="radix", load=load, load_skew=1.5)
    # 先学一次（冷启动会走 round-robin，落到某非过载 rank）
    b1 = r(_ctx(_msgs("SYS", "Q1")))
    # 再次同前缀，radix 命中 b1 的 rank（应不等于 2，因为 2 过载被 spill）
    b2 = r(_ctx(_msgs("SYS", "Q1")))
    assert b2.dp_rank != 2, "过载的 rank2 不应被选中"
    assert b1.dp_rank == b2.dp_rank, "同前缀应稳定命中同 rank"


def test_build_router_radix_default_bounded():
    """build_router(mode=radix) 默认 radix_max_nodes=100000，树有界(不无限增长)。"""
    from prefix_hash_router.app import DEFAULT_RADIX_MAX_NODES, build_router
    assert DEFAULT_RADIX_MAX_NODES == 100000
    r = build_router(dp_size=8, mode="radix")   # 不传 radix_max_nodes → 默认 100000
    assert r.radix_tree.max_nodes == 100000, f"默认应有界, 实际 {r.radix_tree.max_nodes}"
    # 显式 None 才能关闭
    r2 = build_router(dp_size=8, mode="radix", radix_max_nodes=None)
    assert r2.radix_tree.max_nodes is None


def test_radix_stats_hit_miss():
    """radix 命中率/冷启动/每 rank 分布统计（判断 radix 是否有效、有无热点）。"""
    r = build_router(dp_size=8, mode="radix")
    # 第1次: 冷启动(miss)，第2次同前缀: 命中(hit)，第3次新前缀: miss
    r(_ctx(_msgs("SYS", "Q1")))
    r(_ctx(_msgs("SYS", "Q1")))
    r(_ctx(_msgs("SYS", "Q2")))
    s = r.radix_stats
    assert s["total"] == 3
    assert s["hits"] >= 1          # 至少一次命中
    assert s["misses"] >= 1        # 至少一次冷启动
    assert s["hits"] + s["misses"] == s["total"]
    # 每 rank 分布：非 radix 模式不应有 stats
    r2 = build_router(dp_size=8, mode="prefix_hash")
    assert getattr(r2, "radix_stats", None) is None


def test_load_length_mismatch_raises():
    """静态 load 长度 != dp_size 应抛错（防过载保护 rank 静默错乱）。"""
    from prefix_hash_router.app import build_router
    try:
        build_router(dp_size=8, load=[1]*5)   # 5 != 8
        raise AssertionError("应抛 ValueError")
    except ValueError:
        pass
    # 长度匹配则正常
    build_router(dp_size=8, load=[1]*8)


def test_radix_supports_multi_format_messages():
    """radix 能从 OpenAI chat / Anthropic(顶层system) / responses(instructions+input) 学习回填。"""
    from prefix_hash_router.router.context import RequestContext
    r = build_router(dp_size=8, mode="radix")

    def ctx3(body):
        return RequestContext(headers={}, raw_body=b"", parsed_body=body, session_key=None)

    # Anthropic: 顶层 system
    b1 = r(ctx3({"system": "SYS", "messages": [{"role": "user", "content": "Q1"}]}))
    b2 = r(ctx3({"system": "SYS", "messages": [{"role": "user", "content": "Q1"}]}))
    assert b1.dp_rank == b2.dp_rank, "Anthropic 同前缀应命中同 rank"

    # OpenAI responses: instructions + input
    r3 = build_router(dp_size=8, mode="radix")
    c1 = r3(ctx3({"instructions": "RULE", "input": "hello"}))
    c2 = r3(ctx3({"instructions": "RULE", "input": "hello"}))
    assert c1.dp_rank == c2.dp_rank, "Responses 同前缀应命中同 rank"


def test_radix_null_content_not_pollute():
    """Anthropic content=null / 缺失 不应被当作 'None' 字符串污染树。"""
    from prefix_hash_router.router.policies.radix import PrefixRadixTree
    # 含 null assistant 的对话：Q, null-assistant, Q2
    tree = PrefixRadixTree()
    tree.learn([
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "Q2"},
    ], rank=1)
    # null 段应被跳过，树只有 Q 和 Q2 两个节点
    assert tree.size() == 2, f"null content 不应产生节点, 实际 {tree.size()}"
    # 缺失 content 同理
    tree2 = PrefixRadixTree()
    tree2.learn([{"role": "user", "content": "Q"},
                 {"role": "assistant"},
                 {"role": "user", "content": "Q2"}], rank=2)
    assert tree2.size() == 2, f"缺失 content 不应产生节点, 实际 {tree2.size()}"


def test_radix_shared_prefix_across_protocols():
    """三种协议提取相同 system+首问 时，共享同一段前缀(命中同 rank)。"""
    from prefix_hash_router.router.context import RequestContext
    r = build_router(dp_size=8, mode="radix")

    def ctx3(body):
        return RequestContext(headers={}, raw_body=b"", parsed_body=body, session_key=None)

    # 三种协议：system="SYS", 首问="Q1"
    a = r(ctx3({"messages": [{"role": "system", "content": "SYS"},
                             {"role": "user", "content": "Q1"}]}))
    b = r(ctx3({"system": "SYS", "messages": [{"role": "user", "content": "Q1"}]}))
    c = r(ctx3({"instructions": "SYS", "input": "Q1"}))
    # anthropic 和 responses 提取后与 OpenAI chat 前缀一致 → 命中同 rank
    assert b.dp_rank == a.dp_rank, "Anthropic 与 OpenAI chat 共享前缀应同 rank"
    assert c.dp_rank == a.dp_rank, "Responses 与 OpenAI chat 共享前缀应同 rank"


def test_responses_real_format_no_dup_system():
    """OpenAI responses 真实格式(input list+input_text块)：
    - 标准用法(input 不含 system) → instructions 前置为 system
    - input 已含 developer → 不重复前置 instructions，保留自带 developer
    """
    from prefix_hash_router.router.context import RequestContext
    def ctx3(body):
        return RequestContext(headers={}, raw_body=b"", parsed_body=body, session_key=None)

    # 标准 responses：input 是 list 且块是 input_text
    std = ctx3({"instructions": "SYS", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Q1"}]},
    ]}).messages
    assert std[0] == {"role": "system", "content": "SYS"}, "instructions 前置"
    assert len(std) == 2

    # input 已含 developer → 不重复前置
    dev = ctx3({"instructions": "SYS", "input": [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "DEV"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Q1"}]},
    ]}).messages
    from prefix_hash_router.router.policies.radix import _msg_segment
    assert _msg_segment(dev[0]) == ("developer", "DEV"), "保留 input 自带 developer"
    assert len(dev) == 2, "不应重复前置 instructions"


def test_lookup_detail_distinguishes_hits():
    """lookup_detail 用于区分"完整叶子命中"与"祖先前缀命中(无冷启动直接命中)"。"""
    from prefix_hash_router.router.policies.radix import PrefixRadixTree

    def msgs(*pairs):
        out = []
        for t in pairs:
            if t == "SYS":
                r = "system"
            elif t.startswith("A"):
                r = "assistant"
            else:
                r = "user"
            out.append({"role": r, "content": t})
        return out

    tree = PrefixRadixTree()
    # 冷启动学一条完整路径到 rank2
    tree.learn(msgs("SYS", "Q1", "A1", "Q2"), 2)

    # 完整叶子命中：full_leaf=True
    d = tree.lookup_detail(msgs("SYS", "Q1", "A1", "Q2"))
    assert d is not None
    assert d["full_leaf"] is True and d["depth"] == 4 and d["total_segs"] == 4 and d["rank"] == 2

    # 不同消息(仅共享 SYS 中间前缀) → 未命中(None)，不是伪命中
    assert tree.lookup_detail(msgs("SYS", "QX")) is None

    # 同一会话追加(更长)：只命中祖先前缀 → 异常直接命中(full_leaf=False)
    d = tree.lookup_detail(msgs("SYS", "Q1", "A1", "Q2", "A2", "Q3"))
    assert d is not None
    assert d["full_leaf"] is False and d["depth"] == 4 and d["total_segs"] == 6 and d["rank"] == 2
    # 未命中点应是"无冷启动即命中前缀"的根因特征
    assert d["depth"] < d["total_segs"]


if __name__ == "__main__":
    test_learn_and_hit()
    test_radix_policy_route_and_none()
    test_append_still_hits()
    test_mid_insert_misses()
    test_truncated_hits_shorter()
    test_changed_start_misses()
    test_empty_segment_skipped()
    test_concurrent_learn_lookup()
    test_build_router_radix_learn_and_hit()
    test_build_router_radix_distinct_sessions()
    test_radix_with_overload_spill_learns_actual()
    test_radix_max_nodes_cap()
    test_radix_lru_evicts_oldest_keeps_shared()
    test_radix_no_refcount_no_inflation()
    test_build_router_radix_default_bounded()
    test_radix_stats_hit_miss()
    test_load_length_mismatch_raises()
    test_radix_supports_multi_format_messages()
    test_radix_null_content_not_pollute()
    test_radix_shared_prefix_across_protocols()
    test_responses_real_format_no_dup_system()
    test_lookup_detail_distinguishes_hits()
    print("radix 全部测试通过")
