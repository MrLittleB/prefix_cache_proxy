# prefix-hash-router · 前缀感知哈希路由

> 为 vLLM 多 DP 集群实现「前缀感知哈希路由」：把相同前缀/多轮会话钉到同一 DP，
> 提升 KV 前缀缓存命中率。作为反向代理，注入 `X-data-parallel-rank` 头并透明透传。

---

## 一、它能做什么

- **前缀缓存路由**：相同对话前缀/会话 → 同一 DP，命中 vLLM KV 前缀缓存，减少重复计算。
- **多协议支持**：OpenAI Chat / Anthropic Messages / OpenAI Responses 三种消息接口，统一做前缀路由。
- **防热点**：会话键优先 + 过载保护（基于真实 running/waiting 负载），避免共享大前缀把某个 DP 打爆。
- **精确前缀匹配（radix 模式）**：从 `messages[0]` 逐条匹配最长连续前缀，中间插入/截断/改开头都能正确处理；LRU 内存有界。
- **可观测**：radix 命中率/分布统计、实际 engine 数 vs 配置 dp_size 诊断。

---

## 二、三层架构

```
Ingress（接入）→ Router（路由核心/可插拔 Policy）→ Dispatcher（转发/注入 rank 头）→ vLLM 多 DP
```

| 层 | 目录 | 职责 |
|---|---|---|
| Ingress | `src/prefix_hash_router/ingress/` | 收 HTTP、读 body、构造 RequestContext、接口分类 |
| Router | `src/prefix_hash_router/router/` | 纯逻辑：组合 RoutingPolicy，ctx → Backend |
| Dispatcher | `src/prefix_hash_router/dispatcher/` | 注入 `X-data-parallel-rank`、透传 header/body/SSE、断开检测 |

**可插拔策略**（`router/policies/`）：SessionAffinity / PrefixHash / Radix / ConsistentHash / RoundRobin / OverloadGuard

---

## 三、安装与快速启动

```bash
# 1) 安装(dev)
pip install -e .

# 2) 启动：前缀哈希路由（推荐先试这个简单模式）
python main.py --dp-size 8 --mode prefix_hash \
    --upstream "http://localhost:9100/v1" \
    --port 38294

# 3) 精确前缀匹配(radix)：适合大量长对话/多轮缓存，需学习回填
python main.py --dp-size 8 --mode radix \
    --upstream "http://localhost:9100/v1" \
    --port 38294

# 对照：固定 rank 0 / 纯轮转
python main.py --mode first_rank --upstream "http://localhost:9100/v1"
python main.py --mode round_robin --upstream "http://localhost:9100/v1"
```

> **配置来源优先级：命令行参数 > .env > 代码默认**。推荐把真实 `UPSTREAM`/`METRICS_TOKEN` 放 `.env`（不入库），命令行只传临时覆盖项。
> **`--upstream` 默认指向 `http://localhost:9100/v1`（占位符，不再是真实地址）**；`--metrics-url` 默认由 upstream 推导为同 host 的 `/metrics`。

---

## 四、全部 CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `38294` | 监听端口 |
| `--upstream` | 集群地址 | 上游 vLLM base(到 /v1) |
| `--metrics-url` | 由 upstream 推导 | vLLM /metrics 地址(每 DP running/waiting) |
| `--metrics-token` | 无 | 拉 /metrics 的 Bearer token(若 /metrics 需要鉴权) |
| `--dp-size` | `8` | DP 数量（**必须与集群实际 engine 数一致**，启动会诊断）|
| `--mode` | `prefix_hash` | `prefix_hash` / `consistent_hash` / `radix` / `first_rank` / `round_robin` |
| `--waiting-weight` | `4.0` | waiting 的**渐进权重上限**（第1个waiting=2、第2个=3、第3个起=4，避免小waiting被夸大）|
| `--load-skew` | `1.5` | 过载相对失衡系数（配合 min-load 用，默认 1.5）|
| `--min-load-for-overload` | `5` | 过载绝对门槛(running+waiting)：首选 rank 负载≥该值才进入相对失衡判定；低于不干预，保持 radix 前缀一致 |
| `--no-overload` | 关 | 关闭真实 metrics 过载保护（radix 命中即稳定，完全无 spill）|
| `--max-body-size` | `0` 不限制 | 请求体上限(字节)，超限 413 |
| `--max-workers` | `256` | 并发线程上限(防线程爆炸) |
| `--radix-max-nodes` | `100000` | radix 树节点上限，LRU 有界 |
| `--debug-rank` | 关 | 打印每个请求的路由诊断(分配到的rank/radix命中/会话键/radix断点/overload spill) |

---

## 四·一：`.env` 配置（推荐存敏感项）

除命令行外，可在 `.env` 里配置（优先级：**命令行 > .env > 代码默认**）。
`.env` 已被 `.gitignore` 忽略，**不会提交**；真实密钥/地址务必放这里，不要写进代码或命令行历史。

```bash
cp .env.example .env    # 编辑填写真实值
```

支持的键：

| 键 | 对应参数 | 说明 |
|---|---|---|
| `UPSTREAM` | `--upstream` | vLLM 上游 base(到 /v1) |
| `METRICS_URL` | `--metrics-url` | /metrics 地址(默认由 UPSTREAM 推导) |
| `METRICS_TOKEN` | `--metrics-token` | 拉 /metrics 的 Bearer token（敏感）|
| `HOST` / `PORT` | `--host` / `--port` | 监听地址/端口 |
| `DP_SIZE` | `--dp-size` | DP 数量 |
| `MODE` | `--mode` | 路由模式 |
| `WAITING_WEIGHT` | `--waiting-weight` | 渐进权重上限(默认4) |
| `LOAD_SKEW` | `--load-skew` | 相对失衡系数(默认1.5) |
| `MIN_LOAD_FOR_OVERLOAD` | `--min-load-for-overload` | 过载绝对门槛(默认5) |
| `MAX_BODY_SIZE` / `MAX_WORKERS` / `RADIX_MAX_NODES` | 同名参数 | 上限配置 |
| `DEBUG_RANK` | `--debug-rank` | true/false |

---

## 四·补充：`--debug-rank` 详细路由诊断

加 `--debug-rank` 后，每次 chat 请求会打印：
```
[route] POST /v1/chat/completions -> rank=3 radix冷启动(完全未命中 walk=0/1) → argmin rank=3 msgs=1
[route] POST /v1/chat/completions -> rank=0 radix命中(leaf walk=4/4 rank=0) msgs=4  ← 同前缀命中同rank
[route] POST /v1/chat/completions -> rank=0 radix冷启动(前缀占比≤50% walk=1/2) → argmin rank=0 msgs=2
[route] POST /v1/chat/completions -> rank=2 radix命中(子树回退 walk=3/4 subtree_rank=2) final-rank=2 msgs=4
[route] POST /v1/chat/completions -> rank=6 radix命中(leaf walk=1/1 rank=6) session='sess-abc' msgs=1
[route] GET /v1/models -> 透传(非chat, 不加rank)
```
- `radix冷启动(完全未命中 walk=0/N) → argmin rank=X`：树里没有该前缀，选最闲 rank
- `radix冷启动(前缀占比≤50% walk=X/Y) → argmin rank=X`：匹配了浅前缀但占比不足，选最闲 rank
- `radix冷启动(中间断档 walk=X/Y) → argmin rank=X`：深层断档且子树无 rank，选最闲 rank
- `radix命中(leaf walk=N/N rank=X)`：命中叶子节点 → 过载保护
- `radix命中(子树回退 walk=X/Y subtree_rank=R) final-rank=X`：walk>50% 子树 BFS 找到 rank → 过载保护
- `radix命中(末尾追加 walk=X/Y miss=Z rank=A→B)`：对话多轮追加 → 过载保护
- `session='...'`：命中了会话键（会话粘性最强）
- 非 chat 请求显示透传

**新增诊断（`[radix-debug]` 与 `[overload] SPILL`）**：

`[radix-debug]` 打印每次 radix 决策前/后的命中与断点信息，用于定位"为什么命中深度比上次学的少"：
```
[radix-debug] rank=2 msgs条数=15 非空段=13
  PRE walk=13 found=1@13 break_idx=13/13 break_seg=(role, 内容前40字)
  POST walk=13 found=1@13 post_break=-1
```
- `walk`：实际沿树匹配到的段数；`found`：命中的 rank 及深度
- `break_idx/break_seg`：在哪个段断档、该段内容（若 `break_idx` 落在中间 = messages 中间内容变了，会重新随机 fallback → 漂移）

`[overload] SPILL` 打印过载 spill 的触发原因，用于定位"rank 乱跳是否是过载误判"：
```
[overload] SPILL 触发: 首选rank=5 负载=30.00 绝对门槛=5 相对阈值=5.62 (avg=3.75 skew=1.5) 全负载=[0,0,0,0,0,30,0,0]
    -> spill 到 rank=3 (via HRW, atom_key='...')
```
- 只有「首选 rank 负载 ≥ 绝对门槛(5) **且** > avg×1.5」才 spill
- 若日志频繁出现 SPILL 且你认为"这不是真过载"，可调大 `--min-load-for-overload` 或用 `--no-overload`

> 这是验证"相同前缀是否被钉到同一 rank"的最直接手段。

---

## 五、mode 怎么选

| mode | 适用 | 特点 |
|---|---|---|
| `prefix_hash` | 简单、无状态、多实例友好 | 按 system/首条稳定前缀哈希，`hash%dp` |
| `radix` | **大量长对话/多轮缓存/同 agent 分支** | 精确最长前缀匹配；**有状态、仅单实例**、需学习回填、LRU 有界 |
| `consistent_hash` | DP 扩缩容频繁 | 一致性哈希环，最小重映射 |
| `first_rank` / `round_robin` | 对照/调试 | 固定 rank 0 / 纯轮转 |

**建议**：先 `prefix_hash` 跑通、看命中率；如果对话很长且想最大化命中，切 `radix`。

### radix 的关键行为（理解它怎么路由）

radix 树在叶子(完整消息序列终点)标记归属 rank，**中间节点 rank 保留不清除**（支持 agent 回溯重发）：

### 完整路由决策树

```
请求进入
│
├─ 第1层: 有 session key? → HRW → 过载保护 → 结束
│
├─ 第2层: radix 树查找
│   ├─ 1. walk=0 完全未命中 → 找最闲rank (argmin load)
│   ├─ 2. 命中非叶子节点
│   │   ├─ 2.1 walk/total > 50% → 子树BFS找rank → 过载保护
│   │   └─ 2.2 walk/total ≤ 50% → 找最闲rank (argmin load)
│   └─ 3. 命中叶子节点 → found=rank → 过载保护
│
├─ 过载保护(命中rank): load < min_load → 直接用; > avg*skew → spill到相对空闲rank
└─ 找最闲rank(未命中): argmin(load), 多个最闲随机选; 无负载数据 → 随机
```

| 场景 | 路由策略 |
|------|----------|
| **不同会话**(叶子不同，哪怕共享 system) → 分散到不同 rank，各自用独立 DP KV cache |
| **同一会话多轮**(完整历史命中同一叶子) → 粘在同一 rank，命中多轮缓存 |
| **Agent 回溯**(从中间节点重新发问) → 命中中间节点保留的 rank，复用 KV cache |
| **深层分叉**(walk>50% 但非叶子) → 子树回退找 rank → 过载保护 |
| **浅层共享**(walk≤50%，如只共享 system) → argmin 最闲 rank |
| **完全未命中**(walk=0) → argmin 最闲 rank |

### LRU 淘汰策略

- 找最旧叶子 → 从叶子向上删 → 遇到以下任一条件停止：
  - **共享点**(父还有其他孩子) → 停止，保留共享前缀
  - **有 rank 的节点** → 停止，保护有效短对话/agent 回溯点
  - **根节点** → 停止
- 冷的 rank 叶子节点最终也会被淘汰（不会永驻）

---

## 六、支持的接口（接入模型）

### 会加 `X-data-parallel-rank` 头(做前缀路由)—— 3 个消息生成接口
```
POST /v1/chat/completions   (OpenAI Chat)
POST /v1/messages           (Anthropic)
POST /v1/responses          (OpenAI Responses)
```
三种协议的请求体都能被标准化提取为消息序列做前缀匹配，且共享相同前缀时命中同 rank。

### 直接透传(不加 rank, vLLM 自己负载均衡)
```
GET  /v1/models, /health, /metrics, /ping, /load, /version, /docs, /openapi.json
POST /v1/completions, /tokenize, /detokenize, /invocations, /generative_scoring
POST /v1/chat/completions/batch|render|derender, /v1/responses/{id}|/cancel, /messages/count_tokens
GET/HEAD/OPTIONS/PUT/PATCH/DELETE 等
```

---

## 七、接入你的模型

### 7.1 直接经代理调用（OpenAI 格式）
```bash
# 把请求发到代理端口，代理会注入 X-data-parallel-rank 并转发到 vLLM
curl http://localhost:38294/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "你的模型名",
    "messages": [
      {"role": "system", "content": "同一个系统提示"},
      {"role": "user", "content": "问题A"}
    ]
  }'
# 再次发"相同 system + 不同问题"，应命中同一 DP(rank)，缓存命中
curl http://localhost:38294/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "你的模型名", "messages": [{"role":"system","content":"同一个系统提示"},{"role":"user","content":"问题B"}]}'
```

### 7.2 经代理调用（Anthropic 格式，如果你走 /v1/messages）
```bash
curl http://localhost:38294/v1/messages \
  -H "x-api-key: 你的key" -H "Content-Type: application/json" \
  -d '{"model":"你的模型名","system":"同一个系统提示","messages":[{"role":"user","content":"问题A"}]}'
```

### 7.3 经代理调用（OpenAI Responses 格式）
```bash
curl http://localhost:38294/v1/responses \
  -H "Authorization: Bearer 你的key" -H "Content-Type: application/json" \
  -d '{"model":"你的模型名","instructions":"同一个系统提示","input":"你好"}'
```

---

## 八、DSH / A-B 实测

项目自带 `tests/live_ab_test.py` 对接真实 vLLM 做 A/B 对照：
- **baseline**：默认负载均衡(不带 rank)
- **fixrank**：固定 rank(如 0)
- **proxy**：经本代理(前缀哈希)

```bash
# 1) 先启动代理（指向你的 vLLM）
python main.py --dp-size 8 --mode prefix_hash \
    --upstream "http://<vllm-host>:9100/v1" --port 38294

# 2) 通过环境变量提供真实连接信息（live_ab_test.py 已改为从环境变量读取，不再硬编码）
export VLLM_BASE="<vllm-base>/v1"      # 如 https://your-vllm:30079/v1
export VLLM_MODEL="<your-model>"
export VLLM_API_KEY="<your-key>"        # 敏感，勿写进代码/命令历史

# 3) 只跑直连 A/B（baseline vs fixrank）
PYTHONPATH=src python3 tests/live_ab_test.py

# 4) 含经代理对照
PYTHONPATH=src python3 tests/live_ab_test.py --with-proxy --proxy-url http://127.0.0.1:38294/v1 \
    --prefix-file /path/to/你的前缀.txt --times 8
```

> 指标主要看 `usage.prompt_tokens_details.cached_tokens`：`cached>0` 即命中。

---

## 九、测试(单元/端到端)

```bash
# 全部 8 个测试
for t in test_router test_dispatcher test_ingress_e2e test_metrics_collector \
         test_disconnect test_edge_cases test_error test_radix; do
  PYTHONPATH=src python3 tests/$t.py
done
```

---

## 十、文档索引
- **总方案.md** — 问题确认 → 业界调研 → 可行性 → 方案设计(三层) → 审核 → 难点 → 里程碑
- **架构设计.md** — 三层架构详细论证
- **反代审查与待办.md** — 健壮性审查 + 已处理/待办清单

## 当前状态
- ✅ 三层架构 + 可插拔策略 + 多协议支持 + radix + LRU + 接口分类
- ✅ **过载判定两层化**：`min-load-for-overload`(绝对门槛,默认5) + `> avg×load_skew`(相对失衡)，避免"1 vs 0"误判 spill
- ✅ **waiting 渐进权重**：`waiting-weight`(上限,默认4)=第1个2/第2个3/第3个起4，避免小 waiting 被夸大
- ✅ **`.env` 配置** + 命令行优先（不硬编码真实地址/密钥）
- ✅ **转发/诊断增强**：`[forward-debug]`(完整地址+body结构) / `[radix-debug]`(命中/断点) / `[overload] SPILL`
- ✅ 8 个测试文件全部通过
- ⏳ 待办：radix 持久化/TTL、多实例共享树(演进)、上游连接池(性能)
