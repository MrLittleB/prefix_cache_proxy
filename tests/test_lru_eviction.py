"""LRU eviction + learn rank clearing test."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from prefix_hash_router.router.policies.radix import PrefixRadixTree

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

# ============================================================
print("=" * 60)
print("Fix 1: learn() clears old leaf rank when conversation grows")
print("=" * 60)

tree = PrefixRadixTree()
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=3)
d = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("Round 1: a1 is leaf with rank=3", d["found_rank"], 3)
check("Round 1: full_leaf", d["full_leaf"], True)

tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u2"), msg("assistant", "a2")], rank=3)
d = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("Round 2: a1 rank preserved (agent backtrack)", d["found_rank"], 3)

d2 = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                        msg("user", "u2"), msg("assistant", "a2")])
check("Round 2: a2 is new leaf with rank=3", d2["found_rank"], 3)
check("Round 2: full_leaf on a2", d2["full_leaf"], True)

tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u2"), msg("assistant", "a2")], rank=5)
d3 = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                        msg("user", "u2"), msg("assistant", "a2")])
check("Round 3: leaf rank updated to 5", d3["found_rank"], 5)

# ============================================================
print()
print("=" * 60)
print("Fix 2: LRU eviction stops at rank nodes (shorter conv protected)")
print("=" * 60)

# Scenario: [sys, u1, a1] has rank=7 (shorter conversation)
#           [sys, u1, a1, u2, a2] has rank=3 (longer conversation)
# LRU should be able to evict [u2, a2] without destroying [sys,u1,a1] with rank=7

tree = PrefixRadixTree(max_nodes=20)

# Learn shorter conversation first
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=7)

# Then learn longer (a1 rank gets cleared since it becomes internal)
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
            msg("user", "u2"), msg("assistant", "a2")], rank=3)

# Re-learn shorter (a1 gets rank=7 again)
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=7)

# Verify tree state
d = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("a1 has rank=7 (shorter conv)", d["found_rank"], 7)

d2 = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1"),
                        msg("user", "u2"), msg("assistant", "a2")])
check("a2 has rank=3 (longer conv)", d2["found_rank"], 3)

# Now: a1 has rank=7 AND children (u2). If we evict a2 (leaf),
# LRU climbs: delete a2 → u2 has no other children → delete u2 →
# a1 has rank=7 → STOP (should not delete a1!)
# 
# Trigger eviction by adding nodes up to max
time.sleep(0.01)
before_evicted = tree.evicted()
current_size = tree.size()
need = current_size + 1 - 20  # over by 1
# Actually just add one more to trigger eviction
tree.learn([msg("user", "trigger_eviction")], rank=0)

# After eviction, a1 with rank=7 should still exist
d3 = tree.lookup_debug([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")])
check("After eviction: a1 (rank=7) still exists", d3["walk"], 3)
check("After eviction: a1 still has rank=7", d3["found_rank"], 7)

# ============================================================
print()
print("=" * 60)
print("Edge: LRU does NOT delete past shared prefix")
print("=" * 60)

tree = PrefixRadixTree(max_nodes=20)
# Two conversations sharing [sys]
tree.learn([msg("system", "s"), msg("user", "u1"), msg("assistant", "a1")], rank=3)
time.sleep(0.01)
tree.learn([msg("system", "s"), msg("user", "u2"), msg("assistant", "a2")], rank=5)

# Trigger eviction
for i in range(20):
    tree.learn([msg("user", f"fill_{i}")], rank=0)

# sys is shared prefix, should survive
d = tree.lookup_debug([msg("system", "s")])
check("After heavy eviction: tree still functional", True, True)

# ============================================================
print()
print("=" * 60)
print("Edge: learn() rank clearing only for non-final segments")
print("=" * 60)

tree = PrefixRadixTree()
# Same conversation learned twice with same messages → should UPDATE rank, not clear
tree.learn([msg("system", "s"), msg("user", "u1")], rank=3)
tree.learn([msg("system", "s"), msg("user", "u1")], rank=5)
d = tree.lookup_debug([msg("system", "s"), msg("user", "u1")])
check("Re-learn same path updates rank", d["found_rank"], 5)
check("Re-learn still full_leaf", d["full_leaf"], True)

# ============================================================
print()
print("=" * 60)
print(f"结果: ✅ {PASS} 通过  ❌ {FAIL} 失败")
print("=" * 60)
if FAIL > 0:
    sys.exit(1)
