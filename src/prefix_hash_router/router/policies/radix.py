"""RadixPrefixPolicy：基于前缀树(radix tree)的精确最长连续前缀匹配路由。

背景：对话缓存命中的是"从消息开头逐条往后的连续前缀"。固定取 K 条(system+前 N 条)
难以拍定 K；完整历史又因 agent 可能在中间插入/修改/截断而漂移。radix tree 从
messages[0] 逐条沿树匹配"最长连续已归属前缀"，天然只锚定"从头连续稳定的部分"，
中间变化/末尾追加/截断都能正确处理——这是业界最精确的无 tokenizer 字符串级方案
（vLLM 缓存引擎内部、SGLang #2114 cache-aware router 同思路）。

架构说明：
  - 这是**有状态**策略：内部维护一棵 message 级前缀树 + 每段前缀的归属 rank。
  - 需要"学习回填"：当一个请求被路由/转发到某 rank 后，需调用 learn() 把该请求的
    messages 序列归属到最终 rank，后续同前缀请求才能命中。
  - **仅单实例可用**：树是进程内状态；多实例需共享/同步（见 反代审查与待办.md 2.9）。
  - 冷启动：树里没有前缀时 route() 返回 None，由 ChainedPolicy 交给 fallback。

内存管理（权衡决策）：
  - **LRU 分支淘汰**：max_nodes 有界 + 淘汰最久未访问的可安全删除分支，让树自动
    收敛到热会话。用 children 结构判定共享前缀（某节点有多个孩子即共享），
    删冷分支时保留共享前缀，避免误删共享 system。
  - **保留 max_nodes 兜底**：即使淘汰逻辑有偏差，树也不会超过硬上限。
  - TTL 过期与磁盘持久化暂缓（无法抓真实流量调参，收益不确定），已在待办标注。

线程安全：多线程 Ingress 下用 RLock 保护树结构。
"""
from __future__ import annotations

import threading
import time

from ..context import RequestContext
from ..backend import Backend
from ._keys import _msg_text


def _msg_segment(m) -> tuple[str, str]:
    """把一条 message 归一化为 (role, text) 段，作为前缀树的一层 key。"""
    if isinstance(m, dict):
        role = str(m.get("role", "")).strip()
        text = _msg_text(m).strip()
    else:
        role = ""
        text = str(m).strip()
    return (role, text)


class _Node:
    """前缀树节点。每层 key 是一段消息 (role, text)。

    children: dict[segment, _Node]
    rank:    该节点归属的 DP rank（None=未标记）
    last_access: 最近一次命中/学习的时间戳（LRU 依据）
    """

    __slots__ = ("children", "rank", "last_access")

    def __init__(self):
        self.children = {}
        self.rank = None
        self.last_access = time.time()


class PrefixRadixTree:
    """message 级前缀树，带 LRU 分支淘汰。

    沿树从 messages[0] 往下匹配，返回最深已标记归属 rank（无则 None）。
    学习(learn)时沿途标记 rank、刷新 last_access。
    达到 max_nodes 时触发 LRU 淘汰最冷"可安全删除分支"（基于 children 保留共享前缀）。
    """

    def __init__(self, max_nodes: int | None = None):
        self._root = _Node()           # 虚拟根节点（不计入 node_count）
        self._lock = threading.RLock()
        self._max_nodes = max_nodes    # 树节点硬上限；None=不限制（不推荐，会无限增长）
        self._node_count = 0           # 当前节点数（不含虚拟根）
        self._evicted = 0              # 累计淘汰节点数（诊断）

    # ---- 学习：把整段 messages 归属到某 rank ----
    def learn(self, messages: list, rank: int) -> None:
        """插入 messages 序列；**只在叶子(完整路径终点)标记 rank**，中间节点不标。

        为什么只在叶子标记(关键优化，解决 KV cache 挤占)：
          - 每个 DP 的 total KV cache 有限。若沿途中间节点(如共享 system/首问)也标 rank，
            共享前几条的多个会话会全部命中同一 rank → 挤占同一 DP 的 cache，互相顶掉，
            最早会话 cache 被挤没，命中率反而最低、其余 DP cache 浪费。
          - 只在叶子标记后：共享 system 但后续不同的会话，各自学习到不同叶子 → 分散到
            不同 rank，各自用独立 cache 不互挤；同一会话多轮(完整历史命中同一叶子)仍粘一起。
          - 折中：放弃"共享中间前缀的缓存复用"，换来"不同会话利用 8 份独立 cache"，
            对短会话(只共享 system/首问)前者收益低、后者收益高。

        rank 归属随最新实际路由结果更新。若将超过 max_nodes，先 LRU 淘汰腾空间。
        """
        with self._lock:
            segs = [s for s in (_msg_segment(m) for m in messages) if s[1]]
            # 需要新增的节点数
            node = self._root
            need_new = 0
            for seg in segs:
                nxt = node.children.get(seg)
                if nxt is None:
                    need_new += 1
                else:
                    node = nxt
            if self._max_nodes is not None and self._node_count + need_new > self._max_nodes:
                self._evict_lru(needed=self._node_count + need_new - self._max_nodes)

            # 正式插入：中间节点只建节点/刷新 last_access，不标 rank
            # 注意：已有节点的 rank 保留不清除——用户用 agent 时可能从中间节点回溯重发，
            # 保留 rank 使 lookup 能在中间节点命中，复用 KV cache。
            node = self._root
            for seg in segs:
                nxt = node.children.get(seg)
                if nxt is None:
                    nxt = _Node()
                    node.children[seg] = nxt
                    self._node_count += 1
                node = nxt
                node.last_access = time.time()
            # 叶子(完整路径终点)标记归属 rank（覆盖旧值）
            node.rank = rank

    # ---- 查询：求最长已归属前缀的 rank（无则 None）----
    def lookup(self, messages: list) -> int | None:
        """沿树从 messages[0] 匹配；命中节点刷新 last_access；返回最深已标记 rank。"""
        with self._lock:
            d = self._lookup_tracked(messages)
            return d[1] if d else None

    def lookup_detail(self, messages: list) -> dict | None:
        """lookup 的诊断版：返回最深的命中明细（含命中深度 vs 消息段数），
        用于区分"完整叶子命中"与"祖先前缀命中"。

        返回值（未命中返回 None）：
          rank        命中的 rank
          depth       命中的节点深度（树里第几段被标记）
          total_segs  该 messages 归一化后的总段数（过滤空 content 后）
          full_leaf   是否为完整叶子命中（depth == total_segs）
        """
        with self._lock:
            d = self._lookup_tracked(messages)
            if d is None:
                return None
            depth, rank, total_segs = d
            return {
                "rank": rank,
                "depth": depth,
                "total_segs": total_segs,
                "full_leaf": depth == total_segs,
            }

    def _lookup_tracked(self, messages: list) -> tuple[int, int, int] | None:
        """内部实现：返回 (最深命中深度, rank, 消息总段数)，未命中返回 None。

        注意：total_segs 是 messages 归一化后的"全部非空段数"（与树是否包含无关），
        depth 是实际沿树往下走的深度（遇到树里没有的段会提前 break）。
        """
        # 先数 messages 的非空总段数（不依赖树）
        segs = [s for s in (_msg_segment(m) for m in messages) if s[1]]
        total_segs = len(segs)
        node = self._root
        found = None
        found_depth = 0
        walked = 0
        for seg in segs:
            nxt = node.children.get(seg)
            if nxt is None:
                break
            walked += 1
            node = nxt
            node.last_access = time.time()
            if node.rank is not None:
                found = node.rank
                found_depth = walked
        if found is None:
            # 情况2: walked>0 但沿途无 rank → 当共享前缀占比 > 1/2 时,
            # 从最后匹配节点的子树找 rank (使深层分叉能复用 KV cache);
            # 占比 ≤ 1/2 (如只共享 system) → 仍返回 None, 交给 fallback 分散到不同 rank,
            # 避免"共享 system 塌缩到同一 rank"的挤占问题。
            # 阈值 1/2 而非 2/3: 因为 assistant 回复具有非确定性(每次不同), 跨对话的
            # 请求自然停在共享 user 消息处, walk/total 天然 ≤ 50%; 同一对话编辑则 > 50%。
            if walked > 0 and total_segs > 0 and walked / total_segs > 1 / 2:
                subtree_rank = self._find_subtree_rank(node)
                if subtree_rank is not None:
                    return (walked, subtree_rank, total_segs)
            return None
        return (found_depth, found, total_segs)

    def lookup_debug(self, messages: list) -> dict:
        """高细节诊断 lookup：返回命中/断点的完整信息，用于精确定位"为什么命中深度比上次学的少"。

        与 lookup_detail 相比额外增加：
          break_index  沿树匹配时，在第几个非空段找不到 child（即断点位置；-1 表示全部匹配到）
          break_seg    断点处这段消息的 (role, text_前若干字符)；若全部匹配则为 None
          walk_length  实际沿树匹配到的非空段数（可能小于断点，因为断点是"缺的那段"）
        始终返回 dict（不会返回 None），方便打印诊断。
        """
        with self._lock:
            segs = [s for s in (_msg_segment(m) for m in messages) if s[1]]
            total_segs = len(segs)
            node = self._root
            found = None
            found_depth = 0
            walked = 0
            break_index = -1
            break_seg = None
            for idx, seg in enumerate(segs):
                nxt = node.children.get(seg)
                if nxt is None:
                    # 记录断点：这是树里没有的第 idx 段（0 起始）
                    break_index = idx
                    break_seg = (seg[0], seg[1][:40])   # role + 内容前40字符
                    break
                walked += 1
                node = nxt
                node.last_access = time.time()
                if node.rank is not None:
                    found = node.rank
                    found_depth = walked
            # 子树回退：与 _lookup_tracked 一致，当 found=None 且 walked/total_segs > 1/2 时
            # 从最后匹配节点的子树找 rank（使诊断日志与实际路由决策一致）
            subtree_rank = None
            if found is None and walked > 0 and total_segs > 0 and walked / total_segs > 1 / 2:
                subtree_rank = self._find_subtree_rank(node)
            # 如果子树找到了 rank，等效于"找到了 rank"（用于诊断分类）
            effective_rank = found if found is not None else subtree_rank
            effective_depth = found_depth if found is not None else (walked if subtree_rank is not None else 0)
            return {
                "segs": total_segs,
                "walk": walked,
                "found_depth": found_depth,
                "found_rank": found,
                "break_index": break_index,       # -1=全部匹配到
                "break_seg": break_seg,           # None=全部匹配到
                "full_leaf": (found is not None and found_depth == total_segs),
                "subtree_rank": subtree_rank,     # 子树回退找到的 rank (None=未触发/未找到)
                "effective_rank": effective_rank,  # 实际生效的 rank (found 优先, fallback 到 subtree)
                "effective_depth": effective_depth,
            }


    def _find_subtree_rank(self, node: _Node) -> int | None:
        """DFS from node, return first found rank in subtree (breadth-first: shallower preferred)."""
        from collections import deque
        q = deque([node])
        while q:
            n = q.popleft()
            if n.rank is not None:
                return n.rank
            for child in n.children.values():
                q.append(child)
        return None

    # ---- LRU 淘汰：删最久未访问的"叶子会话"，保留共享前缀 ----
    def _evict_lru(self, needed: int = 1) -> int:
        """淘汰约 needed 个最冷叶子会话分支。返回实际删除节点数。

        策略(对齐 vLLM/SGLang radix cache 思路)：
          - 收集所有"叶子"节点(无子节点)，按 last_access 最旧优先。
          - 删除最旧叶子，并沿父链向上回收：
            每删一个节点，若其父删除该子后**变空(无其它孩子)且非根且无 rank**，
            则继续向上删(整条冷链)；
            若父还有其它孩子(共享点)、或父是根、或父有 rank(是某个更短对话的叶子)，
            则停止——从而完整释放一条冷会话链，且绝不动共享前缀和有效短对话。
          - 基于 children 结构判定"共享"，不使用 refcount，从根本上避免多轮追加
            导致共享前缀 refcount 虚高的问题。
          - 中间节点的 rank 保留（支持 agent 回溯重发），LRU 淘汰时遇到
            有 rank 的节点停止向上删除，保护有效短对话。
        """
        with self._lock:
            deleted = 0
            while deleted < needed:
                # DFS 收集叶子及其完整路径([(parent,seg,node),...] 根→叶子)
                leaves = []
                def dfs(node, path):
                    if not node.children:
                        leaves.append((node.last_access, list(path)))
                        return
                    for seg, child in node.children.items():
                        path.append((node, seg, child))
                        dfs(child, path)
                        path.pop()
                dfs(self._root, [])
                if not leaves:
                    break
                leaves.sort(key=lambda x: x[0])         # 最旧优先
                _, path = leaves[0]
                # 从叶子沿路径向上回收空链，直到共享点/带rank节点/根
                for i in range(len(path) - 1, -1, -1):
                    parent, seg, node = path[i]
                    del parent.children[seg]
                    self._node_count -= 1
                    self._evicted += 1
                    deleted += 1
                    # 父是根 → 停止（不能删根）
                    if parent is self._root:
                        break
                    # 父还有其它孩子(共享点) → 停止（不能破坏其他分支）
                    if parent.children:
                        break
                    # 父有 rank → 停止（父是某个更短对话的叶子，不能删）
                    if parent.rank is not None:
                        break
                    # 父变空 → 继续删更上一层(下一轮 path[i-1] 的 node 就是 parent)
            return deleted

    # ---- 调试/诊断 ----
    def size(self) -> int:
        return self._node_count

    def evicted(self) -> int:
        return self._evicted

    @property
    def max_nodes(self):
        return self._max_nodes

    def __contains__(self, messages) -> bool:
        return self.lookup(messages) is not None


class RadixPrefixPolicy:
    """基于 PrefixRadixTree 的精确前缀路由。

    - 若树里已学到 messages 的连续前缀归属，返回该 rank。
    - 否则返回 None，交给 ChainedPolicy 的 fallback（如 RoundRobin / PrefixHash）。
    - 需在转发完成后调用 learn(messages, rank) 回填。
    """

    def __init__(self, dp_size: int, tree: PrefixRadixTree | None = None):
        self._dp = dp_size
        # 若外部不注入树，则自建一棵（便于单测）
        self.tree = tree if tree is not None else PrefixRadixTree()

    def route(self, ctx: RequestContext) -> Backend | None:
        msgs = ctx.messages or []
        if not msgs or not isinstance(msgs, list):
            return None
        rank = self.tree.lookup(msgs)
        if rank is None:
            return None
        return Backend(dp_rank=rank)

    def learn(self, ctx: RequestContext | list, rank: int) -> None:
        """把请求 messages（或直接一个 messages list）归属到最终 rank。"""
        msgs = ctx.messages if isinstance(ctx, RequestContext) else ctx
        if msgs and isinstance(msgs, list):
            self.tree.learn(msgs, rank)

    def __repr__(self) -> str:
        return f"RadixPrefixPolicy(dp={self._dp}, tree_nodes={self.tree.size()})"
