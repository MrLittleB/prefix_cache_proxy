"""加载 .env 文件（当前目录优先，其次项目根/上级目录）。

提供的 key 约定（与 main.py 参数对应）：
  UPSTREAM          --upstream
  METRICS_URL       --metrics-url
  METRICS_TOKEN     --metrics-token
  HOST              --host
  PORT              --port
  DP_SIZE           --dp-size
  MODE              --mode
  WAITING_WEIGHT    --waiting-weight
  LOAD_SKEW         --load-skew
  MAX_BODY_SIZE     --max-body-size
  MAX_WORKERS       --max-workers
  RADIX_MAX_NODES   --radix-max-nodes

优先级：启动命令行参数 > .env > 代码内置默认值。
本模块只负责把 .env 里的值读出来，不覆盖 argparse 已经显式传入的值。
"""
from __future__ import annotations

import os

# 依次查找 .env 的目录：当前工作目录、本文件所在目录、项目根（上一级）。
_SEARCH_DIRS = (
    os.getcwd(),
    os.path.dirname(os.path.abspath(__file__)),
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


def _load_env_file(path: str) -> dict:
    """解析一个简单的 KEY=VALUE 形式的 .env 文件。

    - 忽略空行与 # 开头注释行。
    - 支持 value 两侧空白与可选的引号（单引号/双引号）包裹。
    - 若同一个 key 出现多次，后者覆盖前者。
    """
    result = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # 去掉包裹的引号
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                if key:
                    result[key] = value
    except OSError:
        # 找不到/读不了 .env 属于正常（可能未配置），静默返回空
        pass
    return result


def load_env_values() -> dict:
    """按优先级读取 .env：当前目录 > 本模块目录 > 项目根；合并为单个 dict。

    靠后的目录仅在靠前的目录里不存在该 key 时才填补，保证"就近优先"。
    """
    merged: dict = {}
    for d in _SEARCH_DIRS:
        path = os.path.join(d, ".env")
        if not os.path.isfile(path):
            continue
        data = _load_env_file(path)
        for k, v in data.items():
            merged.setdefault(k, v)   # 已经有的 key 不再覆盖（就近优先）
    return merged


def str_or_none(val: str | None) -> str | None:
    """把 .env 里的空字符串规范成 None（例如 METRICS_TOKEN 未配置时）。"""
    if val is None:
        return None
    val = val.strip()
    return val if val else None


def int_or_default(val: str | None, default: int) -> int:
    """把 .env 字符串转 int；非法/缺失回退 default。"""
    if val is None or not val.strip():
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default


def float_or_default(val: str | None, default: float) -> float:
    if val is None or not val.strip():
        return default
    try:
        return float(val.strip())
    except ValueError:
        return default


def bool_or_default(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default
