"""全面测试完整路由决策树。"""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from prefix_hash_router.router.context import RequestContext
from prefix_hash_router.app import build_router
from prefix_hash_router.router.policies.radix import PrefixRadixTree, RadixPrefixPolicy
from prefix_hash_router.router.policies._hash import hrw_hash_rank

PASS = 0
FAIL = 0

def check(name, actual, expected):
    global PASS, FAIL
    if actual == expected:
        PASS += 1
        print(f"  ✅ {name}: {actual}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: got {actual}, expected {expected}")

def msg(role, content):
    return {"role": role, "content": content}

def ctx(messages, session_key=None):
    return RequestContext(
        headers={"x-session-id": session_key} if session_key else {},
        raw_body=b"",
        parsed_body={"messages": messages},
        session_key=session_key,
    )

# ============================================================
print("=" * 60)
print("第1层: SessionAffinity")
print("=" * 60)

# 1a: 有session key → 直接HRW, 不进radix
r = build_router(dp_size=8, mode="radix")
sk = "test-session-123"
expected_rank = hrw_hash_rank(sk, 8)
b = r(ctx([msg("system", "s"), msg("user", "q")], session_key=sk))
check("有session key → HRW rank", b.dp_rank, expected_rank)

# 同一session key → 稳定同rank
b2 = r(ctx([msg("system", "s"), msg("user", "q2")], session_key=sk))
check("同一session key → 稳定同rank", b2.dp_rank, expected_rank)

# 不同session key → 可能不同rank (至少验证不崩溃)
sk2 = "other-session-456"
b3 = r(ctx([msg("system", "s"), msg("user", "q")], session_key=sk2))
check("不同session key → 不崩溃", isinstance(b3.dp_rank, int) and 0 <= b3.dp_rank < 8, True)

# 1b: 无session key → 进入第2层
b4 = r(ctx([msg("system", "s"), msg("user", "q")]))
check("无session key → 进入radix层", isinstance(b4.dp_rank, int) and 0 <= b4.dp_rank < 8, True)

# ============================================================
print("\n" + "=" * 60)
print("第2层-情况1: 完全未命中 (walk=0) → 找最闲rank")
print("=" * 60)

# 2-1a: 有负载数据 → argmin
load = [5, 2, 8, 1, 4, 6, 3, 7]
r = build_router(dp_size=8, mode="radix", load=load, min_load=0.1)
b = r(ctx([msg("user", "not_in_tree")]))
check("完全未命中 + load=[5,2,8,1,...] → rank3 (load=1最闲)", b.dp_rank, 3)

# 2-1b: 多个最闲 → 随机选一个 (每次用不同消息避免learn-backfill命中radix)
load = [0, 0, 0, 0, 8, 8, 8, 8]
r = build_router(dp_size=8, mode="radix", load=load, min_load=0.1)
results = set()
for i in range(100):
    b = r(ctx([msg("user", f"not_in_tree_{i}")]))  # 每次不同消息,避免radix命中
    results.add(b.dp_rank)
check("完全未命中 + 多个最闲 → 只选最闲的rank", results.issubset({0,1,2,3}), True)
check("完全未命中 + 多个最闲 → 选最小索引(rank集中)", 0 in results, True)

# 2-1c: 无负载数据 → 随机 (每次用不同消息避免learn-backfill)
r = build_router(dp_size=8, mode="radix")
results = set()
for i in range(50):
    b = r(ctx([msg("user", f"cold_start_{i}")]))
    results.add(b.dp_rank)
check("完全未命中 + 无负载 → 选rank=0", results == {0}, True)

# ============================================================
print("\n" + "=" * 60)
print("第2层-情况2.1: walk/total > 50%, 非叶子, 子树BFS → 过载保护")
print("=" * 60)

# 树: [s, u1, a1, u2, a2] → rank=3 (只有a2有rank)
# 请求: [s, u1, a1, X] → walk=3/4=75% > 50% → 子树BFS找rank=3
tree = PrefixRadixTree()
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u2"), msg("assistant", "a2")], rank=3)

# 2.1a: 未过载 → 直接用子树rank
load = [0, 0, 0, 2, 0, 0, 0, 0]  # rank3 load=2 < min_load=5
r = build_router(dp_size=8, mode="radix", load=load, min_load=5.0)
r.radix_tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                     msg("user", "u2"), msg("assistant", "a2")], rank=3)
b = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"), msg("user", "X")]))
check("2.1 未过载 → 子树rank=3", b.dp_rank, 3)

# 2.1b: 过载 → spill
load = [0, 0, 0, 20, 0, 0, 0, 0]  # rank3 load=20, avg=2.5, threshold=3.75
r = build_router(dp_size=8, mode="radix", load=load, load_skew=1.5, min_load=5.0)
r.radix_tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                     msg("user", "u2"), msg("assistant", "a2")], rank=3)
b = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"), msg("user", "X")]))
check("2.1 过载 → spill到其他rank", b.dp_rank != 3, True)
check("2.1 过载spill → 选最闲的", b.dp_rank in [0,1,2,4,5,6,7], True)

# ============================================================
print("\n" + "=" * 60)
print("第2层-情况2.2: walk/total ≤ 50%, 非叶子 → 找最闲rank")
print("=" * 60)

# 请求: [s, X] → walk=1/2=50% ≤ 50% → 找最闲
load = [5, 2, 8, 1, 4, 6, 3, 7]
r = build_router(dp_size=8, mode="radix", load=load, min_load=0.1)
r.radix_tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                     msg("user", "u2"), msg("assistant", "a2")], rank=3)
b = r(ctx([msg("system", "s"), msg("user", "X")]))
check("2.2 walk=50% → 找最闲rank=3 (load=1)", b.dp_rank, 3)

# [s, Y] → walk=1/2=50% → 也是最闲
b2 = r(ctx([msg("system", "s"), msg("user", "Y")]))
check("2.2 另一个walk=50% → 也找最闲rank=3", b2.dp_rank, 3)

# ============================================================
print("\n" + "=" * 60)
print("第2层-情况3: 命中叶子节点 → found=rank → 过载保护")
print("=" * 60)

# 3a: 未过载 → 直接用
load = [0, 0, 0, 2, 0, 0, 0, 0]  # rank3 load=2 < min_load=5
r = build_router(dp_size=8, mode="radix", load=load, min_load=5.0)
r.radix_tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                     msg("user", "u2"), msg("assistant", "a2")], rank=3)
b = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
           msg("user", "u2"), msg("assistant", "a2")]))
check("3 命中叶子 + 未过载 → rank=3", b.dp_rank, 3)

# 3b: 过载 → spill
load = [0, 0, 0, 20, 0, 0, 0, 0]
r = build_router(dp_size=8, mode="radix", load=load, load_skew=1.5, min_load=5.0)
r.radix_tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                     msg("user", "u2"), msg("assistant", "a2")], rank=3)
b = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
           msg("user", "u2"), msg("assistant", "a2")]))
check("3 命中叶子 + 过载 → spill", b.dp_rank != 3, True)

# ============================================================
print("\n" + "=" * 60)
print("过载保护三层判定")
print("=" * 60)

# 测试 min_load / avg*skew 的三层逻辑
# load = [0, 0, 0, 3, 0, 0, 0, 0], min_load=5, skew=1.5
# avg = 3/8 = 0.375, threshold = 0.375 * 1.5 = 0.5625
# rank3 load=3: >= min_load=5? No (3<5) → 第一层通过 → 直接用
load = [0, 0, 0, 3, 0, 0, 0, 0]
r = build_router(dp_size=8, mode="radix", load=load, load_skew=1.5, min_load=5.0)
r.radix_tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=3)
b = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")]))
check("过载保护: load=3 < min_load=5 → 直接用rank=3", b.dp_rank, 3)

# load = [0, 0, 0, 6, 0, 0, 0, 0], min_load=5, skew=1.5
# avg = 6/8 = 0.75, threshold = 0.75 * 1.5 = 1.125
# rank3 load=6: >= min_load=5? Yes. <= threshold=1.125? No (6>1.125) → 过载 → spill
load = [0, 0, 0, 6, 0, 0, 0, 0]
r = build_router(dp_size=8, mode="radix", load=load, load_skew=1.5, min_load=5.0)
r.radix_tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=3)
b = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")]))
check("过载保护: load=6 >= min_load=5 且 > avg*skew → spill", b.dp_rank != 3, True)

# ============================================================
print("\n" + "=" * 60)
print("Agent 真实场景: 多用户 + 过载")
print("=" * 60)

load = [1, 0, 0, 1, 0, 0, 0, 0]  # rank0,3有1个请求
r = build_router(dp_size=8, mode="radix", load=load, min_load=5.0, load_skew=1.5)

# 用户A的对话 (学到rank1)
r(ctx([msg("system", "agent"), msg("user", "天气如何")]))  # 冷启动 → 最闲rank
# 用户B的对话 (学到rank2)
r(ctx([msg("system", "agent"), msg("user", "写代码")]))    # 冷启动 → 最闲rank

# 用户A继续对话 → radix命中 → rank稳定
b = r(ctx([msg("system", "agent"), msg("user", "天气如何")]))
rank_a = b.dp_rank

b2 = r(ctx([msg("system", "agent"), msg("user", "天气如何"), msg("assistant", "晴天"), msg("user", "明天呢")]))
check("用户A追加对话 → 同rank", b2.dp_rank, rank_a)

# 新用户C冷启动 → 找最闲
b3 = r(ctx([msg("system", "agent"), msg("user", "翻译一下")]))
check("用户C冷启动 → 选最闲rank", b3.dp_rank in [1,2,4,5,6,7], True)

# ============================================================
print("\n" + "=" * 60)
print("负载均衡效果验证")
print("=" * 60)

# 8个不同用户冷启动，load初始全0
load = [0]*8
r = build_router(dp_size=8, mode="radix", load=load, min_load=5.0, load_skew=1.5)
rank_counts = {}
for i in range(8):
    b = r(ctx([msg("system", "agent"), msg("user", f"question_{i}")]))
    rank = b.dp_rank
    rank_counts[rank] = rank_counts.get(rank, 0) + 1
    # 模拟load更新: 冷启动后该rank load+1
    load[rank] += 1
    # 重建router以模拟负载变化 (实际环境由MetricsCollector提供)
    r = build_router(dp_size=8, mode="radix", load=list(load), min_load=5.0, load_skew=1.5)

print(f"  8个冷启动分布: {rank_counts}")
check("8冷启动 → 每个rank分到1个", len(rank_counts), 8)
check("8冷启动 → 无rank超载", all(v == 1 for v in rank_counts.values()), True)

# ============================================================
print("\n" + "=" * 60)
print("radix统计: hit/miss")
print("=" * 60)

r = build_router(dp_size=8, mode="radix")
# 冷启动 → miss
r(ctx([msg("system", "s"), msg("user", "q1")]))
# 同前缀 → hit
r(ctx([msg("system", "s"), msg("user", "q1")]))
# 新前缀 → miss
r(ctx([msg("system", "s"), msg("user", "q2")]))

stats = r.radix_stats
check("radix total=3", stats["total"], 3)
check("radix hits≥1", stats["hits"] >= 1, True)
check("radix misses≥1", stats["misses"] >= 1, True)
check("radix hits+misses=total", stats["hits"] + stats["misses"], stats["total"])

# ============================================================
print("\n" + "=" * 60)
print(f"结果: ✅ {PASS} 通过  ❌ {FAIL} 失败")
print("=" * 60)

if FAIL > 0:
    sys.exit(1)
