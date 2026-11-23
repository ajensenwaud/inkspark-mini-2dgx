#!/usr/bin/env python3
"""Markdown tables from spark-bench summary.json files (stdlib only).

  python3 tools/release/table.py decode <summary.json>[=label] ...   one stream: tok/s per workload; 2+: aggregate
  python3 tools/release/table.py depth <summary.json> ...            prompt tokens, first token, prefill, decode
"""
import json
import sys


def decode(paths):
    rows = []
    for p in paths:
        path, _, label = p.partition("=")
        s = json.load(open(path))
        by = {}
        for d in s["decode"]:
            by.setdefault(d["concurrency"], {})[d["workload"]] = d
        for c in sorted(by):
            w = by[c]
            if c == 1:
                cells = [f"{w[k]['per_stream_tps_mean']:.1f}" for k in ("chat", "code", "agent", "report")]
                mean = sum(w[k]["per_stream_tps_mean"] for k in w) / len(w)
                rows.append(f"| {label or s['label']} | 1 | {' / '.join(cells)} | {mean:.1f} per stream |")
            else:
                agg = [w[k]["aggregate_decode_tps"] for k in ("chat", "code", "agent", "report")]
                rows.append(f"| {label or s['label']} | {c} | {' / '.join(f'{a:.0f}' for a in agg)} | "
                            f"{sum(agg) / len(agg):.0f} in total ({sum(w[k]['per_stream_tps_mean'] for k in w) / len(w):.1f} a stream) |")
    print("| run | streams | chat / code / agent / report | mean |\n|---|---|---|---|")
    print("\n".join(rows))


def depth(paths):
    print("| prompt tokens | first token | prefill | decode | samples |\n|---|---|---|---|---|")
    for p in paths:
        s = json.load(open(p))
        for d in s["depth"]:
            print(f"| {d['prompt_tokens']:,} | {d['ttft_s']:.1f} s | {d['prefill_tps']:,.0f} tok/s | "
                  f"{d['decode_tps']:.1f} tok/s | {d['samples']} |")


if __name__ == "__main__":
    {"decode": decode, "depth": depth}[sys.argv[1]](sys.argv[2:])
