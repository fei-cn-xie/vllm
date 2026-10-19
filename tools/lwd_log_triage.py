#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triage LWD bring-up logs: which hop of the pre-recv chain broke?

Reads edge/cloud serve logs (encoding-adaptive: BOM, bare UTF-16, GBK,
mixed line endings - never crashes), reconstructs the per-seqno chain

    edge EMBED -> cloud notify -> worker SUBMIT -> DONE -> REPORT
    -> gate opens (no more "pre-recv" deferrals) -> worker POST

and prints startup health plus a verdict per broken hop, so a hung
request localizes to one segment instead of "cloud spins, edge waits".

Usage:
    python3 tools/lwd_log_triage.py CLOUD.log [EDGE.log ...] [--encoding X] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------- #
# Encoding-adaptive reading (hard requirement; production logs are often
# UTF-16 from PowerShell redirection, with or without BOM)
# ---------------------------------------------------------------------- #
_BOMS: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe", "utf-16"),  # BOM consumed by the codec
    (b"\xfe\xff", "utf-16"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
)


def detect_encoding(path: Path, sample_bytes: int = 64 * 1024) -> str:
    with path.open("rb") as fh:
        head = fh.read(sample_bytes)
    for bom, encoding in _BOMS:
        if head.startswith(bom):
            return encoding
    # Bare UTF-16 heuristic: NUL bytes on one parity of a printable sample.
    sample = head[: sample_bytes // 2]
    if sample and b"\x00" in sample:
        nul = float(sample.count(b"\x00")) / max(len(sample), 1)
        odd = sum(1 for i in range(1, len(sample), 2) if sample[i:i + 1] == b"\x00")
        even = sum(1 for i in range(0, len(sample), 2) if sample[i:i + 1] == b"\x00")
        if nul > 0.125 and (odd == 0) != (even == 0):
            return "utf-16-le" if odd > even else "utf-16-be"
    for encoding in ("utf-8", "gbk", "big5"):
        try:
            head.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"  # caller reads with errors="replace"


def iter_lines(path: Path, encoding: str | None = None) -> list[str]:
    encoding = encoding or detect_encoding(path)
    text = path.read_text(encoding=encoding, errors="replace")
    return [line.rstrip("\n\r") for line in text.splitlines()]


# ---------------------------------------------------------------------- #
# Log grammar (formats taken from the emitting code, not guessed)
# ---------------------------------------------------------------------- #
@dataclass
class Chain:
    """Observable events for one seqno (ts from the shared monotonic clock)."""

    embed: int | None = None          # edge line no.
    notify: int | None = None         # cloud-sched saw the notify
    submit: tuple[float, int] | None = None   # (ts, line no.)
    done: tuple[float, int] | None = None
    report: tuple[float, int] | None = None
    post: int | None = None
    defers: dict[str, int] = field(default_factory=dict)


_PATTERNS = {
    "attached": re.compile(
        r"\[pre-recv\] hint MQ created by engine, done MQ attached: (OK|NONE)"
    ),
    "acquire_failed": re.compile(
        r"\[pre-recv\] done MQ acquire from (\S+) failed \((\w+): ([^)]*)\)"
    ),
    "hint_missing": re.compile(r"\[pre-recv\] hint MQ missing"),
    "comm_thread": re.compile(
        r"\[Lwd\]\[pre-recv\] comm thread starting rank=(\d+) hint MQ=(\w+)"
    ),
    "hint_dropped": re.compile(r"\[Lwd\]\[pre-recv\] HINT DROPPED: (.+)"),
    "submit": re.compile(r"\[Lwd\]\[pre-recv\] SUBMIT channel=\S+ seqno=(\d+).*ts=([\d.]+)"),
    "done": re.compile(r"\[Lwd\]\[pre-recv\] DONE channel=\S+ seqno=(\d+) ts=([\d.]+)"),
    "report": re.compile(r"\[Lwd\]\[pre-recv\] REPORT channel=\S+ seqno=(\d+) ts=([\d.]+)"),
    "post": re.compile(r"\[Lwd\]\[pre-recv\] POST seqno=(\d+) .*endpoint=True"),
    "defer": re.compile(
        r"\[cloud-sched\] notify seqno=(\d+) deferred: (.+?)\s*(?:reqs=|$)"
    ),
    "embed": re.compile(r"\[LWD\]\[edge-worker\] EMBED seqno=(\d+) reqs=(\d+) tokens=(\d+)"),
    "engine_assembled": re.compile(r"\[cloud\] engine assembled: control plane on (\S+)"),
    "edge_assembled": re.compile(r"\[edge\] engine assembled: cloud \S+ at (\S+)"),
    "cloud_params": re.compile(r"\[cloud-params\] scheduling limits verified"),
    "http_post": re.compile(r'"POST /(v1/)?(chat/)?completions'),
    "has_work": re.compile(r"\[has_work\] engines_running=(\w+) has_requests=(\w+)"),
    "channels": re.compile(r"data-plane channels created .*my_rank=(\d+), endpoints=(\S+)"),
    "warmup": re.compile(r"data-plane warmup done: channel=\S+ my_rank=(\d+) peer=(\d+)"),
    "port_down": re.compile(r"Communication_Error_Initialize_Transport|network port is down"),
    "gate_never": re.compile(r"pre-recv gate will never open"),
}

_CLOUD_MARKERS = ("cloud-sched", "[cloud] engine assembled", "comm thread starting",
                  "LwdCloudEngine", "cloud-exec", "hint MQ created by engine",
                  "done MQ attached")
_EDGE_MARKERS = ("edge-worker", "EMBED seqno=", "LwdEdgeEngine",
                 "[edge] engine assembled", "has_work")


@dataclass
class FileReport:
    path: Path
    role: str = "unknown"
    startup: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    chains: dict[int, Chain] = field(default_factory=dict)
    encoding: str = ""
    http_posts: int = 0
    has_pending_requests: bool = False

    def chain(self, seqno: int) -> Chain:
        return self.chains.setdefault(seqno, Chain())


def analyze(path: Path, encoding: str | None) -> FileReport:
    report = FileReport(path=path)
    lines = iter_lines(path, encoding)
    report.encoding = encoding or detect_encoding(path)
    text = "\n".join(lines)
    cloud_hits = sum(text.count(m) for m in _CLOUD_MARKERS)
    edge_hits = sum(text.count(m) for m in _EDGE_MARKERS)
    report.role = "cloud" if cloud_hits > edge_hits else "edge" if edge_hits else "unknown"

    for no, line in enumerate(lines, 1):
        if _PATTERNS["port_down"].search(line):
            report.errors.append(
                f"L{no}: NPU 网口 down(EI0009) - hccn_tool -i <id> optical/ip/lldp 三连排查"
            )
        if _PATTERNS["gate_never"].search(line):
            report.startup.append(f"L{no}: {line.strip()[:120]}")
        m = _PATTERNS["attached"].search(line)
        if m:
            report.startup.append(f"L{no}: done MQ attached: {m.group(1)}")
        m = _PATTERNS["acquire_failed"].search(line)
        if m:
            report.startup.append(
                f"L{no}: done MQ acquire failed at {m.group(1)}: {m.group(2)}: {m.group(3)}"
            )
        m = _PATTERNS["hint_missing"].search(line)
        if m:
            report.startup.append(f"L{no}: hint MQ missing (老代码路径在跑,未同步)")
        m = _PATTERNS["comm_thread"].search(line)
        if m:
            report.startup.append(f"L{no}: comm thread rank={m.group(1)} hint MQ={m.group(2)}")
        m = _PATTERNS["hint_dropped"].search(line)
        if m:
            report.errors.append(f"L{no}: HINT DROPPED {m.group(1)}")
        m = _PATTERNS["engine_assembled"].search(line)
        if m:
            report.startup.append(f"L{no}: control plane on {m.group(1)}")
        m = _PATTERNS["edge_assembled"].search(line)
        if m:
            report.startup.append(f"L{no}: edge assembled, cloud at {m.group(1)}")
        if _PATTERNS["cloud_params"].search(line):
            report.startup.append(f"L{no}: cloud scheduling limits verified (边云握手闭环)")
        if _PATTERNS["http_post"].search(line):
            report.http_posts += 1
        m = _PATTERNS["has_work"].search(line)
        if m and m.group(2) == "True":
            report.has_pending_requests = True
        m = _PATTERNS["warmup"].search(line)
        if m:
            report.startup.append(f"L{no}: warmup my_rank={m.group(1)} peer={m.group(2)}")
        for key in ("submit", "done", "report"):
            m = _PATTERNS[key].search(line)
            if m:
                seqno, ts = int(m.group(1)), float(m.group(2))
                setattr(report.chain(seqno), key, (ts, no))
        m = _PATTERNS["post"].search(line)
        if m:
            report.chain(int(m.group(1))).post = no
        m = _PATTERNS["defer"].search(line)
        if m:
            chain = report.chain(int(m.group(1)))
            chain.notify = chain.notify or no
            reason = m.group(2).strip()
            chain.defers[reason] = chain.defers.get(reason, 0) + 1
        m = _PATTERNS["embed"].search(line)
        if m:
            report.chain(int(m.group(1))).embed = no
    return report


# ---------------------------------------------------------------------- #
# Verdicts: first matching rule names the broken hop (matrix from the
# 2026-10 bring-up; each rule cites the observable it keys on)
# ---------------------------------------------------------------------- #
def verdicts(reports: list[FileReport]) -> list[str]:
    out: list[str] = []
    cloud = next((r for r in reports if r.role == "cloud"), None)
    edge = next((r for r in reports if r.role == "edge"), None)
    if cloud is None:
        out.append("未识别出云侧日志(缺 cloud-sched/[cloud] engine assembled 标记)")
        return out
    joined = " | ".join(cloud.startup) + " | " + " | ".join(cloud.errors)
    if not any("done MQ attached" in s for s in cloud.startup):
        if "hint MQ missing" in joined:
            out.append("[启动] 老引擎代码在跑: vllm 未同步到含引擎侧建 MQ 的提交")
        else:
            out.append("[启动] 云引擎 __init__ 的 pre-recv 日志缺失: vllm 仓未同步或角色分支未生效")
    elif "done MQ attached: NONE" in joined:
        if "acquire failed" in joined and ("ConnectionRefused" in joined or "Timeout" in joined):
            out.append("[启动] wire store 不可达: 边未先起或 29600 未放通(acquire failed 见上)")
        else:
            out.append("[启动] done MQ 未挂上且非连接类失败: worker 发布未执行(ascend 未同步或 init_device 时序)")
    if any("hint MQ=none" in s for s in cloud.startup):
        out.append("[启动] worker 未 attach hint MQ: 环境变量句柄未随 spawn 传递")
    for seqno, chain in sorted(cloud.chains.items()):
        tag = f"[seqno={seqno}]"
        edge_seen = edge is not None and seqno in edge.chains and edge.chains[seqno].embed
        if not chain.notify and not chain.submit and not chain.post:
            if edge_seen:
                out.append(f"{tag} 边已 EMBED 但云无任何事件: 控制面预告未达引擎(ingest/ZMQ)")
            continue
        if chain.notify and not chain.submit and not chain.post:
            out.append(f"{tag} 预告已到引擎,worker 无 SUBMIT: 通信线程未消费 hint MQ")
            continue
        if chain.submit and not chain.done:
            out.append(f"{tag} 已 SUBMIT 未 DONE: UP irecv 未收齐(数据面尺寸/通道)")
            continue
        if chain.done and not chain.report:
            out.append(f"{tag} 已 DONE 未 REPORT: 完成上报未入 done MQ")
            continue
        # The scheduler's pre-recv gate defers silently (no log line), so
        # its signature is: reported, no POST, no logged defer reason.
        if chain.report and not chain.post and not chain.defers:
            out.append(f"{tag} 已 REPORT 无 POST 且无放行日志: 闸门/水位未生效(引擎排水或 is_irecv_complete)")
            continue
        if any("not added" in r for r in chain.defers):
            out.append(f"{tag} 预告到达但请求 ADD 未落地: input_queue 次序问题")
            continue
        if any("preflight" in r for r in chain.defers):
            out.append(f"{tag} 闸门已过但 preflight 拒绝: 预算/名额/KV 调度参数")
            continue
        if chain.post:
            hops = []
            if chain.submit and chain.done:
                hops.append(f"submit→done={chain.done[0] - chain.submit[0]:.3f}s")
            out.append(f"{tag} 链路打点完整(POST 已达前向, {', '.join(hops)}): 查前向/DOWN 回程")
    if not cloud.chains:
        edge_embeds = sorted(
            seqno for seqno, chain in (edge.chains if edge else {}).items()
            if chain.embed
        )
        if edge_embeds:
            out.append(
                f"云日志无任何 seqno 事件,但边已 EMBED seqno={edge_embeds}: "
                "控制面预告未达云引擎(ingest/ZMQ)"
            )
        elif edge is not None and not edge.http_posts:
            out.append("两份日志均无请求期事件,且边侧无 HTTP 请求记录: "
                       "请求未打到边或日志不含请求期(先发请求再抓全量日志)")
        elif edge is not None and edge.http_posts and not edge.has_pending_requests:
            out.append("边收到 HTTP 请求但引擎未消费(has_work 无 pending): "
                       "API server 到 EngineCore 的通路问题")
        elif edge is not None and edge.has_pending_requests and not edge_embeds:
            out.append("边引擎已持有请求但未出 EMBED 批: 边侧调度(picker/EMBED 出批)问题")
        else:
            out.append("云日志中无任何 seqno 事件: 请求未到达云侧(或日志不含请求期)")
    out.extend(f"[错误] {e}" for e in cloud.errors)
    return out


def render(reports: list[FileReport], as_json: bool) -> str:
    if as_json:
        return json.dumps(
            {
                "files": [
                    {
                        "path": str(r.path),
                        "role": r.role,
                        "encoding": r.encoding,
                        "startup": r.startup,
                        "errors": r.errors,
                        "chains": {
                            str(k): {
                                "embed": v.embed, "notify": v.notify, "post": v.post,
                                "submit": v.submit, "done": v.done, "report": v.report,
                                "defers": v.defers,
                            }
                            for k, v in r.chains.items()
                        },
                    }
                    for r in reports
                ],
                "verdicts": verdicts(reports),
            },
            ensure_ascii=False,
            indent=2,
        )
    parts = []
    for r in reports:
        parts.append(f"== {r.path} (role={r.role}, encoding={r.encoding}) ==")
        if r.startup:
            parts.extend(f"  {s}" for s in r.startup)
        else:
            parts.append("  (无启动段标记)")
        parts.append("")
    parts.append("== 结论 ==")
    parts.extend(f"  {v}" for v in verdicts(reports))
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", type=Path, nargs="+", help="edge/cloud serve 日志文件")
    parser.add_argument("--encoding", help="强制指定日志编码(默认自适应)")
    parser.add_argument("--json", action="store_true", help="机读输出")
    args = parser.parse_args(argv)
    reports = []
    for path in args.logs:
        try:
            reports.append(analyze(path, args.encoding))
        except OSError as exc:
            print(f"[triage] 无法读取 {path}: {exc}", file=sys.stderr)
    if not reports:
        return 1
    print(render(reports, args.json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
