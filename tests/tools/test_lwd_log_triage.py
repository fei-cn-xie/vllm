# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Table-driven tests for tools/lwd_log_triage.py.

Encoding cases mirror the production-log reality (UTF-16 with/without
BOM, GBK, CRLF); verdict cases are one synthetic log per broken hop of
the 2026-10 bring-up chain, each asserting the hop the tool must name.
Log-line formats are taken from the emitting code (lwd_cloud_worker /
lwd_cloud_scheduler), not invented.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import lwd_log_triage as triage  # noqa: E402

CLOUD_STARTUP_OK = [
    "(EngineCore pid=1) INFO [lwd_cloud_engine.py:60] [pre-recv] hint MQ "
    "created by engine, done MQ attached: OK",
    "(Worker pid=2) INFO [lwd_cloud_worker.py:360] [Lwd][pre-recv] comm "
    "thread starting rank=0 hint MQ=attached",
]
EMBED = ("(EngineCore pid=3) INFO [lwd_edge_worker.py:76] [lwd] [LWD]"
         "[edge-worker] EMBED seqno=0 reqs=1 tokens=11 has_mrope=[False]")


def _run(tmp_path: Path, cloud_lines: list[str], edge_lines: list[str] | None = None):
    reports = []
    cloud = tmp_path / "cloud.log"
    cloud.write_text("\n".join(cloud_lines) + "\n", encoding="utf-8")
    reports.append(triage.analyze(cloud, None))
    if edge_lines is not None:
        edge = tmp_path / "edge.log"
        edge.write_text("\n".join(edge_lines) + "\n", encoding="utf-8")
        reports.append(triage.analyze(edge, None))
    return triage.verdicts(reports)


# ---------------- encoding ----------------
@pytest.mark.parametrize(
    "payload, encoding",
    [
        ("EMBED seqno=0 正常", "utf-8"),
        ("EMBED seqno=0 bom", "utf-8"),
        ("EMBED seqno=0 utf16", "utf-16"),
    ],
)
def test_detect_encoding_bom_variants(tmp_path: Path, payload: str, encoding: str) -> None:
    raw = {"utf-8": payload.encode("utf-8"),
           "utf-8-sig": b"\xef\xbb\xbf" + payload.encode("utf-8"),
           "utf-16": b"\xff\xfe" + payload.encode("utf-16-le")}.get(encoding)
    if encoding == "utf-8":
        raw = payload.encode("utf-8")
    path = tmp_path / "enc.log"
    path.write_bytes(raw)
    detected = triage.detect_encoding(path)
    assert detected.startswith(encoding.replace("-sig", ""))


def test_detect_encoding_bare_utf16le_and_gbk(tmp_path: Path) -> None:
    le = tmp_path / "le.log"
    le.write_bytes("EMBED seqno=0 中文内容拉长样本".encode("utf-16-le"))
    assert triage.detect_encoding(le) == "utf-16-le"
    gbk = tmp_path / "gbk.log"
    gbk.write_bytes("EMBED seqno=0 中文内容".encode("gbk"))
    assert triage.detect_encoding(gbk) == "gbk"


def test_iter_lines_never_crashes_and_strips_crlf(tmp_path: Path) -> None:
    path = tmp_path / "crlf.log"
    path.write_bytes("line one\r\nline two\r\n".encode("utf-16"))
    lines = triage.iter_lines(path, None)
    assert lines == ["line one", "line two"]


# ---------------- role + verdict table ----------------
def test_role_detection(tmp_path: Path) -> None:
    cloud = tmp_path / "c.log"
    cloud.write_text("[cloud-sched] notify seqno=0 deferred: x\n", encoding="utf-8")
    assert triage.analyze(cloud, None).role == "cloud"
    edge = tmp_path / "e.log"
    edge.write_text(EMBED + "\n", encoding="utf-8")
    assert triage.analyze(edge, None).role == "edge"


VERDICT_CASES = [
    (
        "engine_code_absent",
        [
            "(Worker pid=9) INFO [lwd_cloud_worker.py:166] [lwd] [Lwd][perf] "
            "cloud-exec total=0.13ms",
        ],
        None,
        "vllm 仓未同步",
    ),
    (
        "attached_none_connection",
        CLOUD_STARTUP_OK[:0] + [
            "[pre-recv] hint MQ created by engine, done MQ attached: NONE",
            "[pre-recv] done MQ acquire from tcp://1.2.3.4:29600 failed "
            "(ConnectionRefusedError: [Errno 111] refused)",
        ],
        None,
        "wire store 不可达",
    ),
    (
        "attached_none_no_publish",
        [
            "[pre-recv] hint MQ created by engine, done MQ attached: NONE",
        ],
        None,
        "worker 发布未执行",
    ),
    (
        "hint_mq_none",
        [
            "[pre-recv] hint MQ created by engine, done MQ attached: OK",
            "[Lwd][pre-recv] comm thread starting rank=0 hint MQ=none",
        ],
        None,
        "环境变量句柄",
    ),
    (
        "notify_never_reached_cloud",
        CLOUD_STARTUP_OK,
        [EMBED],
        "控制面预告未达",
    ),
    (
        "submit_missing",
        CLOUD_STARTUP_OK + [
            "[cloud-sched] notify seqno=0 deferred: request(s) not added yet",
        ],
        None,
        "通信线程未消费",
    ),
    (
        "done_missing",
        CLOUD_STARTUP_OK + [
            "[Lwd][pre-recv] SUBMIT channel=lwd_up seqno=0 num_tokens=11 "
            "aux_rows=0 inflight=1 queued=0 ts=100.5",
        ],
        None,
        "UP irecv 未收齐",
    ),
    (
        "report_missing",
        CLOUD_STARTUP_OK + [
            "[Lwd][pre-recv] SUBMIT channel=lwd_up seqno=0 num_tokens=11 "
            "aux_rows=0 inflight=1 queued=0 ts=100.5",
            "[Lwd][pre-recv] DONE channel=lwd_up seqno=0 ts=100.9",
        ],
        None,
        "完成上报未入",
    ),
    (
        "gate_stuck_silent",
        CLOUD_STARTUP_OK + [
            "[Lwd][pre-recv] SUBMIT channel=lwd_up seqno=0 num_tokens=11 "
            "aux_rows=0 inflight=1 queued=0 ts=100.5",
            "[Lwd][pre-recv] DONE channel=lwd_up seqno=0 ts=100.9",
            "[Lwd][pre-recv] REPORT channel=lwd_up seqno=0 ts=100.95",
        ],
        None,
        "闸门/水位未生效",
    ),
    (
        "add_not_landed",
        CLOUD_STARTUP_OK + [
            "[Lwd][pre-recv] SUBMIT channel=lwd_up seqno=0 num_tokens=11 "
            "aux_rows=0 inflight=1 queued=0 ts=100.5",
            "[Lwd][pre-recv] DONE channel=lwd_up seqno=0 ts=100.9",
            "[Lwd][pre-recv] REPORT channel=lwd_up seqno=0 ts=100.95",
            "[cloud-sched] notify seqno=0 deferred: request(s) not added yet",
        ],
        None,
        "ADD 未落地",
    ),
    (
        "preflight_rejected",
        CLOUD_STARTUP_OK + [
            "[Lwd][pre-recv] SUBMIT channel=lwd_up seqno=0 num_tokens=11 "
            "aux_rows=0 inflight=1 queued=0 ts=100.5",
            "[Lwd][pre-recv] DONE channel=lwd_up seqno=0 ts=100.9",
            "[Lwd][pre-recv] REPORT channel=lwd_up seqno=0 ts=100.95",
            "[cloud-sched] notify seqno=0 deferred: preflight rejected "
            "reqs=1 tokens=11",
        ],
        None,
        "preflight",
    ),
    (
        "chain_complete",
        CLOUD_STARTUP_OK + [
            "[Lwd][pre-recv] SUBMIT channel=lwd_up seqno=0 num_tokens=11 "
            "aux_rows=0 inflight=1 queued=0 ts=100.5",
            "[Lwd][pre-recv] DONE channel=lwd_up seqno=0 ts=100.9",
            "[Lwd][pre-recv] REPORT channel=lwd_up seqno=0 ts=100.95",
            "[Lwd][pre-recv] POST seqno=0 reqs=1 tokens=11 endpoint=True "
            "ts=101.2 mem_alloc=1.00GiB",
        ],
        None,
        "链路打点完整",
    ),
]


@pytest.mark.parametrize("name,cloud,edge,expected", VERDICT_CASES,
                         ids=[c[0] for c in VERDICT_CASES])
def test_verdict_names_the_broken_hop(
    tmp_path: Path, name: str, cloud: list[str], edge: list[str] | None, expected: str
) -> None:
    verdicts = _run(tmp_path, cloud, edge)
    assert any(expected in v for v in verdicts), f"{name}: {verdicts}"


def test_chain_hop_deltas_rendered(tmp_path: Path) -> None:
    verdicts = _run(tmp_path, VERDICT_CASES[-1][1], None)
    assert any("submit→done=0.400s" in v for v in verdicts), verdicts
