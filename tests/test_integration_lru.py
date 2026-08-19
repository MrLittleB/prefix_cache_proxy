"""Full integration test: radix tree + LRU eviction + agent backtracking."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from prefix_hash_router.router.policies.radix import PrefixRadixTree
from prefix_hash_router.app import build_router
from prefix_hash_router.router.context import RequestContext
from prefix_hash_router.router.policies._hash import hrw_hash_rank

PASS = FAIL = 0
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
print("1. Agent回溯: 中间节点rank保留, 可命中KV cache")
print("=" * 60)

tree = PrefixRadixTree()
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=3)
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u2"), msg("assistant", "a2")], rank=3)

d = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("a1 rank保留=3 (中间节点, 支持回溯)", d["found_rank"], 3)

d2 = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                         msg("user", "u2"), msg("assistant", "a2")])
check("a2 rank=3 (叶子)", d2["found_rank"], 3)

# 用户回溯: 从a1重新开始发不同的问题
d3 = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                         msg("user", "X")])
check("回溯 walk=3/4>50%, found_rank=3", d3["found_rank"], 3)
check("回溯 effective_rank=3", d3["effective_rank"], 3)

# ============================================================
print()
print("=" * 60)
print("2. LRU淘汰: 向上回溯遇rank节点停止")
print("=" * 60)

tree = PrefixRadixTree(max_nodes=100)

# 场景: [sys,u1,a1] rank=7 (短对话), [sys,u1,a1,u2,a2] rank=3 (长对话)
# a1.rank=7 保留(中间节点)
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=7)
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u2"), msg("assistant", "a2")], rank=3)

# 添加大量冷节点触发淘汰, 但用lookup保持[sys,u1,a1]热
for i in range(200):
    tree.learn([msg("user", f"cold_{i}")], rank=0)
    # 每10次访问一次a1, 保持它热
    if i % 10 == 0:
        tree.lookup([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])

# a1.rank=7 应该存活 (热 + LRU遇rank停止)
d = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("热节点a1存活 (walk=3)", d["walk"], 3)
check("热节点a1.rank=7", d["found_rank"], 7)

# ============================================================
print()
print("=" * 60)
print("3. LRU淘汰: 冷rank节点最终也会被淘汰(不会永驻)")
print("=" * 60)

tree = PrefixRadixTree(max_nodes=10)
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=7)
time.sleep(0.02)
# 填满到超限, a1是冷叶子, 应该被淘汰
for i in range(20):
    tree.learn([msg("user", f"fill_{i}")], rank=0)

d = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("冷rank叶子最终被淘汰 (walk=0)", d["walk"], 0)
check("冷rank叶子被淘汰 (found_rank=None)", d["found_rank"], None)

# ============================================================
print()
print("=" * 60)
print("4. LRU淘汰: 向上回溯时, 父有rank则停止(不删父)")
print("=" * 60)

# 关键测试: a1有rank=7, a1的子分支被淘汰时, 回溯到a1应停止
tree = PrefixRadixTree(max_nodes=100)

# 构造: [sys] → [u1](rank=7) → [a1] → [u2] → [a2](rank=3)
# u1 有rank=7 (曾经是短对话的叶子)
tree.learn([msg("system", "s"), msg("user", "u1")], rank=7)
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u2"), msg("assistant", "a2")], rank=3)

# 用lookup保持u1热, 但a2冷
time.sleep(0.02)
# 保持u1热
for _ in range(5):
    tree.lookup([msg("system", "s"), msg("user", "u1")])

# 添加冷节点淘汰a2分支
for i in range(200):
    tree.learn([msg("user", f"cold_{i}")], rank=0)
    if i % 10 == 0:
        tree.lookup([msg("system", "s"), msg("user", "u1")])

# u1.rank=7应该存活 (热 + LRU回溯遇rank停止)
d = tree.lookup_debug([msg("system", "s"), msg("user", "u1")])
check("u1存活 (walk=2)", d["walk"], 2)
check("u1.rank=7存活", d["found_rank"], 7)

# a2应该被淘汰 (冷叶子)
d2 = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("a1分支已淘汰 (walk<=2)", d2["walk"] <= 2, True)

# ============================================================
print()
print("=" * 60)
print("5. 完整路由: agent回溯 → 命中中间rank → 过载保护")
print("=" * 60)

load = [1, 1, 1, 1, 1, 1, 1, 1]
r = build_router(dp_size=8, mode="radix", load=load, min_load=5.0, load_skew=1.5)

b1 = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")]))
rank_1 = b1.dp_rank
check(f"第1轮冷启动 → rank={rank_1}", 0 <= rank_1 < 8, True)

b2 = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
             msg("user", "u2"), msg("assistant", "a2")]))
check(f"第2轮末尾追加 → 同rank={rank_1}", b2.dp_rank, rank_1)

b3 = r(ctx([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
             msg("user", "X")]))
check(f"回溯 → 命中a1.rank={rank_1} → 过载保护", b3.dp_rank, rank_1)

# ============================================================
print()
print("=" * 60)
print("6. LRU淘汰: 共享前缀(多孩子)不被删除")
print("=" * 60)

tree = PrefixRadixTree(max_nodes=100)

# 两个对话共享 [sys, u1, a1]
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u2"), msg("assistant", "a2")], rank=3)
time.sleep(0.02)
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u3"), msg("assistant", "a3")], rank=5)

# a1 是共享节点 (有2个孩子u2,u3), 淘汰a2时应停在a1
# 保持a3分支热, a2分支冷
time.sleep(0.02)
for i in range(200):
    tree.learn([msg("user", f"cold_{i}")], rank=0)
    if i % 10 == 0:
        # 保持a3分支热
        tree.lookup([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                      msg("user", "u3"), msg("assistant", "a3")])

# a3分支应存活
d3 = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                         msg("user", "u3"), msg("assistant", "a3")])
check("a3分支存活 (walk=5)", d3["walk"], 5)

# a1作为共享节点应存活
d = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("共享节点a1存活 (walk>=3)", d["walk"] >= 3, True)

# ============================================================
print()
print("=" * 60)
print(f"结果: ✅ {PASS} 通过  ❌ {FAIL} 失败")
print("=" * 60)
if FAIL > 0:
    sys.exit(1)
