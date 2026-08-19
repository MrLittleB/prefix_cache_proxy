"""MetricsCollector：从 vLLM /metrics 按 engine(DP) 拉取真实排队/运行负载。

vLLM 每个 DP 独立暴露 `vllm:num_requests_running{engine=N}` 与
`vllm:num_requests_waiting{engine=N}` 两个 gauge（也已确认本集群 8 个 engine 各一份）。
负载口径：load[i] = running[i] + waiting_score(waiting[i])。

waiting 采用**渐进递增权重**，避免"第 1~2 个 waiting 被高倍放大"造成误判：
  waiting_score(w)：第1个权重 = max_w/2，第2个 = 3*max_w/4，第 3 个起 = max_w。
  以 max_w=4 为例：w=1→2, w=2→5, w=3→9, w=4→13, w=5→17 ...（排队越深权重越高，但前少量不夸大）

用法：
    mc = MetricsCollector(metrics_url, dp_size=8, waiting_weight=4.0)
    load = mc.sample()      # -> [float]*8，running + 渐进weighted waiting
"""
from __future__ import annotations

import re
import ssl
import time
import threading
import urllib.request


def progressive_waiting_score(w: float, max_w: float = 4.0) -> float:
    """把 waiting 数映射为排队压力分（渐进递增）。

    第1个 waiting 权重 max_w/2，第2个 3*max_w/4，第 3 个及以后 = max_w。
    这样前 1~2 个 waiting 的权重较低（不夸大瞬时小波动），
    排队越深每个 waiting 的权重越高（体现真正的积压压力）。
    """
    if w <= 0:
        return 0.0
    if w < 1:
        return max_w / 2 * w          # 小数 waiting（0<w<1）按比例
    score = max_w / 2.0                # 第1个
    if w >= 2:
        score += 3.0 * max_w / 4.0     # 第2个
    if w > 2:
        score += (w - 2) * max_w       # 第3个起每个
    return score


class MetricsCollector:
    # 形如: vllm:num_requests_running{engine="0",model_name="..."} 0.0
    _GAUGE_RE = re.compile(
        r'^vllm:num_requests_(running|waiting)\{([^}]*)\}\s+([0-9.eE+-]+)\s*$'
    )

    def __init__(self, metrics_url: str, dp_size: int = 8,
                 waiting_weight: float = 4.0, refresh_interval: float = 0.5,
                 timeout: float = 10.0, tls_verify: bool = False,
                 auth_token: str | None = None):
        self._url = metrics_url
        self._dp = dp_size
        self._w_weight = waiting_weight   # 作为渐进权重上限(max_w)
        self._interval = refresh_interval
        self._timeout = timeout
        self._verify = tls_verify
        self._token = auth_token
        self._running = [0.0]*dp_size
        self._waiting = [0.0]*dp_size
        self._has_data = False   # 是否成功解析过至少一次(区分"真实全0"与"从未取到数据")
        self._consec_fails = 0     # 连续失败次数
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._opener = self._build_opener()
        # 诊断：实际观察到的 engine 集合 与 因 idx>=dp_size 被丢弃的 engine（发现 dp 配置错误）
        self._observed_engines: set[int] = set()
        self._dropped_engines: set[int] = set()

    def _build_opener(self):
        """构建**不走环境代理**的 opener，确保直连 vLLM metrics。

        环境里常有 http_proxy/https_proxy（如 127.0.0.1:7897），urllib 默认会读环境代理
        把外部 metrics 请求劫持走——多一跳且依赖本地代理存活。这里显式禁用代理，并配好 SSL。
        """
        ctx = ssl.create_default_context()
        if not self._verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        handlers = [
            urllib.request.ProxyHandler({}),  # 空 dict -> 不走任何代理
            urllib.request.HTTPSHandler(context=ctx),
        ]
        return urllib.request.build_opener(*handlers)

    # ---- 解析 ----
    def _parse(self, text: str):
        running = [0.0]*self._dp
        waiting = [0.0]*self._dp
        for line in text.splitlines():
            m = self._GAUGE_RE.match(line.strip())
            if not m:
                continue
            kind, attrs, value = m.group(1), m.group(2), float(m.group(3))
            em = re.search(r'engine="(\d+)"', attrs)
            if not em:
                continue
            idx = int(em.group(1))
            with self._lock:
                if idx >= self._dp:
                    # 实际 engine 数 > 配置 dp_size → 这些 DP 永远不会被路由到（发现配置错误）
                    self._dropped_engines.add(idx)
                else:
                    self._observed_engines.add(idx)
            if idx >= self._dp:
                continue
            if kind == "running":
                running[idx] = value
            else:
                waiting[idx] = value
        with self._lock:
            self._running, self._waiting = running, waiting
            self._has_data = True

    # ---- 诊断：实际 engine 数 vs 配置 dp_size ----
    def diagnosis(self) -> dict:
        """返回实际观察到的 engine 信息，用于发现 dp_size 配置错误。

        例如 dp_size=8 但实际 engine 数=12：
          {'observed': {0..7}, 'dropped': {8..11}, 'observed_dp': 8, 'actual_dp': 12, 'mismatch': True}
        """
        with self._lock:
            observed = set(self._observed_engines)
            dropped = set(self._dropped_engines)
        actual_max = (max(observed) + 1) if observed else 0
        actual_from_dropped = (max(dropped) + 1) if dropped else 0
        actual_dp = max(actual_max, actual_from_dropped)
        return {
            "observed": sorted(observed),
            "dropped": sorted(dropped),
            "configured_dp": self._dp,
            "actual_dp": actual_dp,
            "mismatch": actual_dp > self._dp,
        }

    # ---- 抓取一次（同步） ----
    def fetch_once(self) -> list[float]:
        req = urllib.request.Request(self._url, headers={
            "User-Agent": "prefix-hash-router/0.1",
        })
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        with self._opener.open(req, timeout=self._timeout) as resp:
            self._parse(resp.read().decode("utf-8", "replace"))
        return self.load()

    # ---- 当前负载（running + 渐进加权 waiting） ----
    def load(self) -> list[float] | None:
        """返回各 DP 负载：running[i] + progressive_waiting_score(waiting[i])；
        若从未成功拉取到数据则返回 None（而非误当"全0空闲"）。"""
        with self._lock:
            if not self._has_data:
                return None
            return [self._running[i] + progressive_waiting_score(self._waiting[i], self._w_weight)
                    for i in range(self._dp)]

    def has_data(self) -> bool:
        with self._lock:
            return self._has_data

    # ---- 后台轮询 ----
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        def _loop():
            while not self._stop.is_set():
                try:
                    self.fetch_once()
                    self._consec_fails = 0  # 成功则重置
                except Exception as e:
                    # 拉取失败保留上次值，但连续失败3次则清空数据退化为随机
                    self._consec_fails += 1
                    import sys
                    sys.stderr.write(f"{time.strftime('[%Y-%m-%d %H:%M:%S]')} [metrics] fetch failed: {e} (consecutive={self._consec_fails})\n")
                    if self._consec_fails >= 3:
                        with self._lock:
                            self._has_data = False
                        sys.stderr.write(f"{time.strftime('[%Y-%m-%d %H:%M:%S]')} [metrics] 3 consecutive failures, clearing load data (fallback to random)\n")
                self._stop.wait(self._interval)
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
