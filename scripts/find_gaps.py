#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find idle gaps (bubbles) in Ascend torch_npu.profiler kernel_details.csv.

The script scans consecutive tasks on the same (Step Id, Stream ID) and reports
the time between the end of one task and the start of the next one:

    gap_us = next.start - (prev.start + prev.duration)

It can also compute device-wide idle gaps by merging all task intervals per
step, which is useful for multi-stream timelines. Finally it aggregates gaps
by "fingerprint" (prev_name -> next_name) so repeated patterns are easy to spot.

Only Python standard library is required.

Examples
--------
    python scripts/find_gaps.py --prof-dir /path/to/prof/rank0 --top 30
    python scripts/find_gaps.py --kernel-csv .../kernel_details.csv --steps 3,4
    python scripts/find_gaps.py --prof-dir ... --mode both --json out.json

Output fields use milliseconds for readability; input CSV is in microseconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime

REQUIRED_COLUMNS = [
    "Step Id",
    "Stream ID",
    "Name",
    "Type",
    "Start Time(us)",
    "Duration(us)",
]

PSEUDO_STREAMS = {"", "N/A", "NA", "NONE", "-"}


def _norm_field(name: str) -> str:
    return name.strip().lstrip("﻿")


def _to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NAN", "NONE", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_steps(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                out.extend(str(i) for i in range(int(lo), int(hi) + 1))
            except ValueError:
                out.append(part)
        else:
            out.append(part)
    return out or None


def _parse_list(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    values = {v.strip() for v in raw.split(",") if v.strip()}
    return values or None


def find_kernel_csv(prof_dir: str, rank: int | None) -> str:
    matches = []
    for root, _dirs, files in os.walk(prof_dir):
        if "kernel_details.csv" in files:
            matches.append(os.path.join(root, "kernel_details.csv"))
    if rank is not None:
        rank_pattern = re.compile(rf"(?:^|[/_-])rank{rank}(?=[/_.-]|$)")
        filtered = [p for p in matches if rank_pattern.search(p)]
        if filtered:
            matches = filtered
    if not matches:
        raise SystemExit(f"[error] no kernel_details.csv found under: {prof_dir}")
    if len(matches) > 1:
        listing = "\n  ".join(matches)
        raise SystemExit(
            "[error] found multiple kernel_details.csv files; "
            "pass --kernel-csv or --rank to choose one:\n  " + listing
        )
    return matches[0]


def read_rows(kernel_csv: str) -> tuple[list[dict], list[str]]:
    with open(kernel_csv, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise SystemExit("[error] kernel_details.csv has no header")
        normalized = [_norm_field(x) for x in reader.fieldnames]
        original_map = {norm: orig for norm, orig in zip(normalized, reader.fieldnames)}
        missing = [c for c in REQUIRED_COLUMNS if c not in original_map]
        if missing:
            raise SystemExit(
                "[error] kernel_details.csv missing required columns: "
                f"{missing}\navailable columns: {normalized}"
            )

        rows: list[dict] = []
        for raw in reader:
            row = {norm: (raw.get(orig) or "") for norm, orig in original_map.items()}
            step = row["Step Id"].strip()
            stream = row["Stream ID"].strip()
            name = row["Name"].strip()
            type_ = row["Type"].strip()
            acc = row.get("Accelerator Core", "").strip()
            start = _to_float(row["Start Time(us)"])
            duration = _to_float(row["Duration(us)"])
            if not name or start is None or duration is None or duration < 0:
                continue
            rows.append(
                {
                    "step": step,
                    "stream": stream,
                    "name": name,
                    "type": type_,
                    "accelerator": acc,
                    "start": start,
                    "duration": duration,
                    "end": start + duration,
                }
            )
    return rows, normalized


def _is_numeric_stream(stream: str) -> bool:
    return bool(re.fullmatch(r"\d+", stream))


def _step_sort_key(step: str):
    try:
        return (0, int(step))
    except ValueError:
        return (1, step)


def _stream_sort_key(stream: str):
    try:
        return (0, int(stream))
    except ValueError:
        return (1, stream)


def infer_steps(rows: list[dict]) -> tuple[list[dict], dict[str, tuple[float, float]]]:
    """Assign steps to rows with an empty Step Id using step time windows.

    Communication rows often have an empty Step Id. For device-wide union it is
    useful to place them into the step whose [min_start, max_end] covers their
    start time.
    """
    ranges: dict[str, tuple[float, float]] = {}
    by_step = defaultdict(list)
    for row in rows:
        if row["step"]:
            by_step[row["step"]].append(row)
    for step, items in by_step.items():
        ranges[step] = (
            min(item["start"] for item in items),
            max(item["end"] for item in items),
        )

    if not ranges:
        return rows, ranges

    out = []
    for row in rows:
        if row["step"]:
            out.append(row)
            continue

        start = row["start"]
        candidates = [
            (step, lo, hi)
            for step, (lo, hi) in ranges.items()
            if lo <= start <= hi
        ]
        if candidates:
            step = min(candidates, key=lambda x: abs(x[1] - start))[0]
        else:
            step = min(ranges, key=lambda s: min(abs(start - ranges[s][0]), abs(start - ranges[s][1])))
        row = dict(row)
        row["step"] = step
        out.append(row)
    return out, ranges


def compute_stream_gaps(
    rows: list[dict],
    steps: set[str] | None,
    streams: set[str] | None,
    min_gap_us: float,
    match_prev: re.Pattern | None,
    match_next: re.Pattern | None,
) -> list[dict]:
    refs: dict[str, float] = {}
    grouped = defaultdict(list)
    for row in rows:
        if steps and row["step"] not in steps:
            continue
        if not _is_numeric_stream(row["stream"]):
            continue
        if streams and row["stream"] not in streams:
            continue
        grouped[(row["step"], row["stream"])].append(row)
        refs[row["step"]] = min(refs.get(row["step"], row["start"]), row["start"])

    gaps = []
    for (step, stream), items in grouped.items():
        items.sort(key=lambda r: (r["start"], r["end"]))
        for prev, nxt in zip(items, items[1:]):
            gap_us = nxt["start"] - prev["end"]
            if gap_us < min_gap_us:
                continue
            if match_prev and not match_prev.search(prev["name"]):
                continue
            if match_next and not match_next.search(nxt["name"]):
                continue
            ref = refs[step]
            gaps.append(
                {
                    "step": step,
                    "stream": stream,
                    "gap_ms": round(gap_us / 1000.0, 3),
                    "prev_end_ms": round((prev["end"] - ref) / 1000.0, 3),
                    "next_start_ms": round((nxt["start"] - ref) / 1000.0, 3),
                    "prev_name": prev["name"],
                    "next_name": nxt["name"],
                    "prev_type": prev["type"],
                    "next_type": nxt["type"],
                }
            )
    gaps.sort(key=lambda g: (-g["gap_ms"], _step_sort_key(g["step"]), g["stream"]))
    return gaps


def compute_device_gaps(
    rows: list[dict],
    steps: set[str] | None,
    min_gap_us: float,
    match_prev: re.Pattern | None,
    match_next: re.Pattern | None,
) -> list[dict]:
    by_step = defaultdict(list)
    for row in rows:
        if steps and row["step"] not in steps:
            continue
        by_step[row["step"]].append(row)

    gaps = []
    for step, items in by_step.items():
        items.sort(key=lambda r: (r["start"], r["end"]))
        ref = min(item["start"] for item in items)
        merged = []  # [start, end, start_name, end_name]
        for item in items:
            if merged and item["start"] <= merged[-1][1]:
                if item["end"] > merged[-1][1]:
                    merged[-1][1] = item["end"]
                    merged[-1][3] = item["name"]
            else:
                merged.append([item["start"], item["end"], item["name"], item["name"]])

        for prev, nxt in zip(merged, merged[1:]):
            gap_us = nxt[0] - prev[1]
            if gap_us < min_gap_us:
                continue
            if match_prev and not match_prev.search(prev[3]):
                continue
            if match_next and not match_next.search(nxt[2]):
                continue
            gaps.append(
                {
                    "step": step,
                    "stream": "ALL",
                    "gap_ms": round(gap_us / 1000.0, 3),
                    "prev_end_ms": round((prev[1] - ref) / 1000.0, 3),
                    "next_start_ms": round((nxt[0] - ref) / 1000.0, 3),
                    "prev_name": prev[3],
                    "next_name": nxt[2],
                    "prev_type": "",
                    "next_type": "",
                }
            )
    gaps.sort(key=lambda g: (-g["gap_ms"], _step_sort_key(g["step"])))
    return gaps


def compute_window_gaps(
    rows: list[dict],
    steps: set[str] | None,
    streams: set[str] | None,
    min_gap_us: float,
    prev_re: re.Pattern,
    next_re: re.Pattern,
) -> list[dict]:
    """Measure last <prev> end -> first <next> start per step.

    This is useful when a stage boundary has a few small tasks in between; the
    consecutive-task gap is not the same as the stage-to-stage window.
    """
    by_step = defaultdict(list)
    for row in rows:
        if steps and row["step"] not in steps:
            continue
        if streams and row["stream"] not in streams:
            continue
        if not _is_numeric_stream(row["stream"]):
            continue
        by_step[row["step"]].append(row)

    gaps = []
    for step, items in by_step.items():
        ref = min(item["start"] for item in items)
        prev_candidates = [item for item in items if prev_re.search(item["name"])]
        next_candidates = [item for item in items if next_re.search(item["name"])]
        if not prev_candidates or not next_candidates:
            continue

        best_key = None
        best_pair = None
        for prev in prev_candidates:
            for nxt in next_candidates:
                if nxt["start"] < prev["end"]:
                    continue
                key = (prev["end"], -nxt["start"])
                if best_key is None or key > best_key:
                    best_key = key
                    best_pair = (prev, nxt)
        if best_pair is None:
            continue

        prev, nxt = best_pair
        gap_us = nxt["start"] - prev["end"]
        if gap_us < min_gap_us:
            continue
        gaps.append(
            {
                "step": step,
                "stream": prev["stream"] if prev["stream"] == nxt["stream"] else f"{prev['stream']}->{nxt['stream']}",
                "gap_ms": round(gap_us / 1000.0, 3),
                "prev_end_ms": round((prev["end"] - ref) / 1000.0, 3),
                "next_start_ms": round((nxt["start"] - ref) / 1000.0, 3),
                "prev_name": prev["name"],
                "next_name": nxt["name"],
                "prev_type": prev["type"],
                "next_type": nxt["type"],
            }
        )
    gaps.sort(key=lambda g: (-g["gap_ms"], _step_sort_key(g["step"])))
    return gaps


def aggregate_patterns(gaps: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for gap in gaps:
        grouped[(gap["prev_name"], gap["next_name"])].append(gap)

    patterns = []
    for (prev_name, next_name), items in grouped.items():
        values = [item["gap_ms"] for item in items]
        values_sorted = sorted(values)
        p90_index = min(len(values_sorted) - 1, int(round(0.9 * (len(values_sorted) - 1))))
        patterns.append(
            {
                "prev_name": prev_name,
                "next_name": next_name,
                "count": len(items),
                "total_gap_ms": round(sum(values), 3),
                "median_gap_ms": round(statistics.median(values), 3),
                "p90_gap_ms": round(values_sorted[p90_index], 3),
                "max_gap_ms": round(max(values), 3),
                "steps": sorted({item["step"] for item in items}, key=_step_sort_key),
                "streams": sorted({item["stream"] for item in items}),
            }
        )
    patterns.sort(key=lambda p: (-p["total_gap_ms"], -p["count"]))
    return patterns


def print_table(gaps: list[dict], top: int, title: str) -> None:
    print(f"\n{title}")
    if not gaps:
        print("  (no gaps matched)")
        return
    header = f"{'step':>6} {'stream':>6} {'gap_ms':>9} {'prev_end_ms':>12} {'next_start_ms':>13}  prev -> next"
    print("  " + header)
    print("  " + "-" * len(header))
    for gap in gaps[:top]:
        prev_name = gap["prev_name"]
        next_name = gap["next_name"]
        if len(prev_name) > 42:
            prev_name = prev_name[:39] + "..."
        if len(next_name) > 42:
            next_name = next_name[:39] + "..."
        print(
            f"  {gap['step']:>6} {gap['stream']:>6} {gap['gap_ms']:>9.1f} "
            f"{gap['prev_end_ms']:>12.1f} {gap['next_start_ms']:>13.1f}  {prev_name} -> {next_name}"
        )


def print_patterns(patterns: list[dict], top: int, title: str = "Pattern summary (prev -> next):") -> None:
    print(f"\n{title}")
    if not patterns:
        print("  (no patterns)")
        return
    header = f"{'count':>5} {'total_ms':>10} {'median_ms':>10} {'p90_ms':>10}  prev -> next"
    print("  " + header)
    print("  " + "-" * len(header))
    for pattern in patterns[:top]:
        prev_name = pattern["prev_name"]
        next_name = pattern["next_name"]
        if len(prev_name) > 36:
            prev_name = prev_name[:33] + "..."
        if len(next_name) > 36:
            next_name = next_name[:33] + "..."
        print(
            f"  {pattern['count']:>5} {pattern['total_gap_ms']:>10.1f} "
            f"{pattern['median_gap_ms']:>10.1f} {pattern['p90_gap_ms']:>10.1f}  {prev_name} -> {next_name}"
        )


def write_csv(
    path: str,
    stream_gaps: list[dict],
    device_gaps: list[dict],
    window_gaps: list[dict] | None = None,
) -> None:
    fields = [
        "kind", "step", "stream", "gap_ms", "prev_end_ms", "next_start_ms",
        "prev_name", "next_name", "prev_type", "next_type",
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        groups = [("stream", stream_gaps), ("device", device_gaps), ("window", window_gaps or [])]
        for kind, gaps in groups:
            for gap in gaps:
                row = dict(gap)
                row["kind"] = kind
                writer.writerow({field: row.get(field, "") for field in fields})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Find idle gaps between consecutive tasks in Ascend kernel_details.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prof-dir", help="Profiler output directory; kernel_details.csv is searched recursively")
    source.add_argument("--kernel-csv", help="Path to kernel_details.csv")
    parser.add_argument("--rank", type=int, help="When --prof-dir contains multiple ranks, pick rank<N>")
    parser.add_argument("--steps", help="Comma-separated step ids, e.g. 3,4,5 or 3-5")
    parser.add_argument("--streams", help="Comma-separated stream ids, e.g. 2,3")
    parser.add_argument("--min-gap-ms", type=float, default=1.0, help="Minimum gap in milliseconds")
    parser.add_argument("--top", type=int, default=20, help="Print top N gaps/patterns")
    parser.add_argument("--mode", choices=["stream", "device", "both"], default="stream")
    parser.add_argument("--match-prev", help="Regex filter on previous task name")
    parser.add_argument("--match-next", help="Regex filter on next task name")
    parser.add_argument(
        "--window-prev",
        help="Regex for stage-boundary start task; with --window-next, measures "
             "last <prev> end -> first <next> start per step (even if other tasks are in between)",
    )
    parser.add_argument("--window-next", help="Regex for stage-boundary end task")
    parser.add_argument("--no-infer-step", action="store_true", help="Do not infer empty Step Id from time windows")
    parser.add_argument("--json", dest="json_path", help="Write JSON report")
    parser.add_argument("--csv", dest="csv_path", help="Write CSV report")
    args = parser.parse_args(argv)

    kernel_csv = args.kernel_csv or find_kernel_csv(args.prof_dir, args.rank)
    rows, columns = read_rows(kernel_csv)
    if not rows:
        raise SystemExit("[error] no valid kernel rows found (check required columns and values)")

    if not args.no_infer_step:
        rows, _ = infer_steps(rows)

    steps = _parse_steps(args.steps)
    streams = _parse_list(args.streams)
    step_filter = set(steps) if steps else None
    min_gap_us = max(0.0, args.min_gap_ms) * 1000.0
    match_prev = re.compile(args.match_prev) if args.match_prev else None
    match_next = re.compile(args.match_next) if args.match_next else None
    if bool(args.window_prev) != bool(args.window_next):
        raise SystemExit("[error] --window-prev and --window-next must be used together")
    window_prev = re.compile(args.window_prev) if args.window_prev else None
    window_next = re.compile(args.window_next) if args.window_next else None

    window_gaps = []
    if window_prev and window_next:
        window_gaps = compute_window_gaps(
            rows, step_filter, streams, min_gap_us, window_prev, window_next
        )

    stream_gaps = []
    device_gaps = []
    if args.mode in ("stream", "both"):
        stream_gaps = compute_stream_gaps(
            rows, step_filter, streams, min_gap_us, match_prev, match_next
        )
    if args.mode in ("device", "both"):
        device_gaps = compute_device_gaps(rows, step_filter, min_gap_us, match_prev, match_next)

    patterns = aggregate_patterns(stream_gaps) if stream_gaps else []
    device_patterns = aggregate_patterns(device_gaps) if args.mode in ("device", "both") and device_gaps else []

    available_steps = sorted({row["step"] for row in rows}, key=_step_sort_key)
    print("=" * 80)
    print("Ascend kernel gap report")
    print("=" * 80)
    print(f"kernel_csv      : {kernel_csv}")
    print(f"columns         : {len(columns)}")
    print(f"valid rows      : {len(rows)}")
    print(f"steps           : {', '.join(available_steps) if available_steps else '(none)'}")
    print(f"mode            : {args.mode}")
    print(f"min_gap_ms      : {args.min_gap_ms}")
    print(f"step filter     : {', '.join(sorted(step_filter, key=_step_sort_key)) if step_filter else '(all)'}")
    print(f"stream filter   : {', '.join(sorted(streams, key=_stream_sort_key)) if streams else '(all numeric)'}")

    if window_gaps:
        print_table(
            window_gaps,
            args.top,
            f"Window gaps ({args.window_prev} -> {args.window_next}):",
        )
    if args.mode in ("stream", "both"):
        print_table(stream_gaps, args.top, "Top stream gaps:")
    if args.mode in ("device", "both"):
        print_table(device_gaps, args.top, "Top device-wide idle gaps:")
    print_patterns(patterns, args.top)
    if device_patterns:
        print_patterns(device_patterns, args.top, "Device-wide pattern summary (prev -> next):")

    if args.json_path:
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "kernel_csv": kernel_csv,
            "mode": args.mode,
            "min_gap_ms": args.min_gap_ms,
            "steps": available_steps,
            "filters": {
                "steps": sorted(step_filter, key=_step_sort_key) if step_filter else None,
                "streams": sorted(streams, key=_stream_sort_key) if streams else None,
                "match_prev": args.match_prev,
                "match_next": args.match_next,
            },
            "window_gaps": window_gaps,
            "stream_gaps": stream_gaps,
            "device_gaps": device_gaps,
            "patterns": patterns,
            "device_patterns": device_patterns,
        }
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\n[json] {args.json_path}")

    if args.csv_path:
        write_csv(args.csv_path, stream_gaps, device_gaps, window_gaps)
        print(f"[csv ] {args.csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
