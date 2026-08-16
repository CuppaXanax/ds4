#!/usr/bin/env python3
"""Summarize DS4_VULKAN_TIMELINE_JSON into a compact stage heatmap.

The JSON contains host recording events and Vulkan timestamp intervals.  This
tool deliberately keeps attribution conservative: GPU idle is only reported
between dispatches with valid absolute timestamp pairs, while submit/wait and
descriptor time remain CPU-side events.
"""
import argparse
import csv
import json
import math
from collections import defaultdict


def _int_field(event, key):
    value = event.get(key, 0)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float) and math.isfinite(value):
        return max(int(value), 0)
    return 0


def _stage_name(event):
    value = event.get("stage")
    return value if isinstance(value, str) and value else "unassigned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("timeline", help="DS4_VULKAN_TIMELINE_JSON output")
    ap.add_argument("--csv", help="write stage CSV")
    args = ap.parse_args()
    try:
        with open(args.timeline, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        ap.error(f"cannot read timeline JSON: {exc}")
    if not isinstance(doc, dict) or not isinstance(doc.get("events", []), list):
        ap.error("timeline JSON must contain an events array")
    events = doc.get("events", [])
    events = [e for e in events if isinstance(e, dict)]
    rows = defaultdict(lambda: defaultdict(int))
    dispatches = [e for e in events if e.get("kind") == "dispatch"]
    last_end = 0
    for e in dispatches:
        stage = _stage_name(e)
        rows[stage]["gpu_ns"] += _int_field(e, "gpu_ns")
        rows[stage]["dispatches"] += 1
        start = _int_field(e, "gpu_start_ns")
        end = _int_field(e, "gpu_end_ns")
        if start and end >= start:
            if last_end and start > last_end:
                rows[stage]["gpu_idle_ns"] += start - last_end
            last_end = max(last_end, end)
    for e in events:
        kind = e.get("kind")
        stage = _stage_name(e)
        if kind == "barrier":
            rows[stage]["barrier_events"] += 1
        elif kind == "submit":
            rows[stage]["submit_ns"] += _int_field(e, "duration_ns")
        elif kind == "wait":
            rows[stage]["wait_ns"] += _int_field(e, "duration_ns")
        if kind == "descriptor_alloc":
            rows[stage]["descriptor_allocs"] += 1
    fields = ("stage", "dispatches", "gpu_ms", "idle_ms", "barriers",
              "submit_ms", "wait_ms", "descriptor_allocs")
    print("stage                 dispatches gpu_ms  idle_ms barriers submit_ms wait_ms desc")
    print("--------------------  ---------- ------- ------- -------- --------- -------- ----")
    output = []
    for stage, r in sorted(rows.items(), key=lambda kv: -kv[1]["gpu_ns"]):
        line = {
            "stage": stage,
            "dispatches": r["dispatches"],
            "gpu_ms": r["gpu_ns"] / 1e6,
            "idle_ms": r["gpu_idle_ns"] / 1e6,
            "barriers": r["barrier_events"],
            "submit_ms": r["submit_ns"] / 1e6,
            "wait_ms": r["wait_ns"] / 1e6,
            "descriptor_allocs": r["descriptor_allocs"],
        }
        output.append(line)
        display_stage = stage.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        print(f"{display_stage[:20]:20} {line['dispatches']:10d} {line['gpu_ms']:7.3f}"
              f" {line['idle_ms']:7.3f} {line['barriers']:8d}"
              f" {line['submit_ms']:9.3f} {line['wait_ms']:8.3f}"
              f" {line['descriptor_allocs']:4d}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for line in output:
                writer.writerow(line)


if __name__ == "__main__":
    main()
