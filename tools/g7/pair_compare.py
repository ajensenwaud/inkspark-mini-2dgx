#!/usr/bin/env python3
"""Paired comparison of two arms on the same frozen items (stdlib only): McNemar per task and pooled over tasks.

  python3 tools/g7/pair_compare.py <dir> <tagA> <tagB> [--tasks gsm8k] [--json out.json]

LingSpark's tools/eval/pair_compare.py (from the quantised-pack-quality-gate skill's pair_compare.py). Re-parses every saved reply with eval_tasks.parse (a
parser fix rescores both arms), keeps the ids both arms answered, and reports per task: accuracy on those ids, the
discordant counts (only A right / only B right), McNemar z = (only_B - only_A) / sqrt(only_A + only_B) with its
two-sided normal p, and the exact binomial p (report it when fewer than ~25 items are discordant); truncated and
unparsed counts and generated tokens per arm. "pooled" sums the discordant counts over tasks (stratified McNemar).

G7 reading (A = bf16 KV, B = fp8 KV, the recipe default): fp8 "measurably lowers" scores when the exact p < 0.05 with
B below A.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tasks import correct, parse  # noqa: E402


def load(path: Path, task: str) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if "error" in r:
            continue
        got = parse(task, r["text"])
        out[r["id"]] = {"ok": correct(task, got, r["gold"]), "trunc": r.get("finish") == "length",
                        "unparsed": got is None, "tokens": r.get("tokens") or 0}
    return out


def exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p: binomial(n = b + c, 0.5), doubled smaller tail, capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def stats(only_a: int, only_b: int) -> dict:
    n = only_a + only_b
    z = (only_b - only_a) / math.sqrt(n) if n else 0.0
    return {"mcnemar_z": round(z, 3), "p_normal": round(math.erfc(abs(z) / math.sqrt(2)), 4),
            "p_exact": round(exact_p(only_a, only_b), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("tag_a")
    ap.add_argument("tag_b")
    ap.add_argument("--tasks", default="gsm8k,mmlu_pro")
    ap.add_argument("--json", help="also write the result here")
    a = ap.parse_args()
    d, A, B = Path(a.dir), a.tag_a, a.tag_b
    rows, pa, pb, pn = [], 0, 0, 0
    for task in a.tasks.split(","):
        fa, fb = d / f"{A}-{task}.jsonl", d / f"{B}-{task}.jsonl"
        if not (fa.exists() and fb.exists()):
            print(f"{task}: missing {fa if not fa.exists() else fb}", file=sys.stderr)
            continue
        ra, rb = load(fa, task), load(fb, task)
        ids = sorted(i for i in ra if i in rb)
        if not ids:
            print(f"{task}: no items answered by both arms", file=sys.stderr)
            continue
        only_a = sum(ra[i]["ok"] and not rb[i]["ok"] for i in ids)
        only_b = sum(rb[i]["ok"] and not ra[i]["ok"] for i in ids)
        row = {"task": task, "items": len(ids),
               A: {"acc": round(sum(ra[i]["ok"] for i in ids) / len(ids), 4), "correct": sum(ra[i]["ok"] for i in ids),
                   "truncated": sum(ra[i]["trunc"] for i in ids), "unparsed": sum(ra[i]["unparsed"] for i in ids),
                   "gen_tokens": sum(ra[i]["tokens"] for i in ids)},
               B: {"acc": round(sum(rb[i]["ok"] for i in ids) / len(ids), 4), "correct": sum(rb[i]["ok"] for i in ids),
                   "truncated": sum(rb[i]["trunc"] for i in ids), "unparsed": sum(rb[i]["unparsed"] for i in ids),
                   "gen_tokens": sum(rb[i]["tokens"] for i in ids)},
               "delta_points_B_minus_A": round(100 * (sum(rb[i]["ok"] for i in ids) - sum(ra[i]["ok"] for i in ids))
                                               / len(ids), 2),
               f"only_{A}": only_a, f"only_{B}": only_b, **stats(only_a, only_b)}
        rows.append(row)
        pa, pb, pn = pa + only_a, pb + only_b, pn + len(ids)
        print(json.dumps(row))
    if not rows:
        sys.exit("nothing to compare")
    pooled = {"task": "pooled", "items": pn, f"only_{A}": pa, f"only_{B}": pb, **stats(pa, pb)}
    worse = pb < pa and pooled["p_exact"] < 0.05
    pooled["reading"] = (f"{B} measurably worse than {A} (pooled exact p {pooled['p_exact']})" if worse else
                         f"no measurable loss for {B} vs {A} (pooled exact p {pooled['p_exact']})")
    print(json.dumps(pooled))
    if a.json:
        Path(a.json).write_text(json.dumps({"a": A, "b": B, "tasks": rows, "pooled": pooled}, indent=1))


if __name__ == "__main__":
    main()
