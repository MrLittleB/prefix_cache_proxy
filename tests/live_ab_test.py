"""对接真实 vLLM 的 A/B 实测（M4）。

用法:
  PYTHONPATH=src python3 tests/live_ab_test.py            # 只跑直连的 A/B
  PYTHONPATH=src python3 tests/live_ab_test.py --proxy-port 38294   # 含走代理

对照:
  1. baseline  直连默认负载均衡（不带 rank 头）
  2. fixrank   直连固定 X-data-parallel-rank: 0
  3. proxy     经 prefix_hash_router 代理（前缀哈希路由）
指标: cached_tokens>0 比例；同一前缀多次连发的命中稳定性。
"""
import sys, os, json, time, argparse, urllib3, requests
urllib3.disable_warnings()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from prefix_hash_router.router.policies._hash import hash_to_rank

# 敏感信息从环境变量读取，禁止硬编码真实密钥/地址入库。
# 用法: VLLM_BASE=... VLLM_MODEL=... VLLM_API_KEY=... PYTHONPATH=src python3 tests/live_ab_test.py
BASE = os.environ.get("VLLM_BASE", "http://localhost:9100/v1")
MODEL = os.environ.get("VLLM_MODEL", "your-model-name")
KEY = os.environ.get("VLLM_API_KEY", "")


def send(url, system, user, rank=None):
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    if rank is not None:
        headers["X-data-parallel-rank"] = str(rank)
    body = {"model": MODEL, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], "max_tokens": 8, "temperature": 0}
    r = requests.post(url + "/chat/completions", headers=headers, json=body, verify=False, timeout=180)
    try:
        u = r.json()["usage"]
        pt = u.get("prompt_tokens")
        ct = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    except Exception:
        pt = ct = None
    return r.status_code, pt, ct


def run_prefix_series(url, system, user, times, rank=None, label=""):
    hits = 0
    first = True
    print(f"  [{label}] 同前缀连发{times}次 系统前缀字符数={len(system)}")
    for i in range(times):
        st, pt, ct = send(url, system, f"{user}#{i}""", rank)
        hit = (ct or 0) > 0
        if hit: hits += 1
        print(f"    req{i+1}: HTTP {st} prompt_tokens={pt} cached={ct} {'HIT' if hit else 'MISS'}")
        time.sleep(0.15)
    print(f"  => {label} 命中 {hits}/{times} = {hits/times*100:.0f}%")
    return hits/times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proxy-url", default="http://127.0.0.1:38294/v1")
    ap.add_argument("--with-proxy", action="store_true", help="是否测试经代理")
    ap.add_argument("--prefix-file", default="/workspace/ds_playground/prefix_cache_proxy/pfx_C.txt")
    ap.add_argument("--times", type=int, default=8)
    args = ap.parse_args()

    system = open(args.prefix_file).read().strip()
    # 用全新前缀避免缓存污染：这里 pfx_C 之前送过，为公平用 pfx_C+额外唯一标记
    marker = f"\n【AB测试标记】{int(time.time())}"
    prefix = system + marker
    print(f"测试前缀总字符数 ≈ {len(prefix)}")

    print("\n========= A/B: 同一(带唯一标记的)前缀，各模式连发 =========")
    # 1) baseline 默认负载均衡
    run_prefix_series(BASE, prefix, "baseline", args.times, rank=None, label="baseline默认负载均衡")
    # 2) 固定 rank 0
    run_prefix_series(BASE, prefix, "fixrank", args.times, rank=0, label="固定rank=0")
    # 3) 经代理（前缀哈希路由）
    if args.with_proxy:
        run_prefix_series(args.proxy_url, prefix, "proxy", args.times, rank=None, label="经代理(前缀哈希)")

    print("\n========= 多前缀分散验证(经代理) =========")
    if args.with_proxy:
        pfx_file = args.prefix_file
        base_text = open(pfx_file).read().strip()[:2000]
        ranks_via = {}
        for i in range(32):
            sys2 = base_text + f"\n多前缀{i}"
            # 直接经代理
            st, pt, ct = send(args.proxy_url, sys2, f"q{i}", None)
            # 计算它应该落到哪个 rank（用同一哈希逻辑）
            from prefix_hash_router.router.policies._keys import extract_prefix_key
            from prefix_hash_router.router.context import RequestContext
            ctx = RequestContext(headers={}, raw_body=b"", parsed_body={
                "messages": [{"role": "system", "content": sys2}]})
            key = extract_prefix_key(ctx)
            r = hash_to_rank(key, 8)
            ranks_via[i] = r
            time.sleep(0.05)
        from collections import Counter
        dist = Counter(ranks_via.values())
        print("  32 个不同前缀经代理后的 rank 分布:", dict(sorted(dist.items())))
        print("  命中分布是否从集中到分散（>1 个 rank 即为有效分散）")


if __name__ == "__main__":
    main()
