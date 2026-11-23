#!/usr/bin/env python3
"""Freeze 500 Global-MMLU-Lite test items, the one task the Inkling 2 Small card reports that is a plain public
multiple-choice set (91.0 % there; the Mini card reports none of these tasks). The card does not say which languages it averaged, so the sample is drawn from the
union of every language's test split: random.Random(1234).sample over (language, row) pairs sorted by language,
then sorted. Output items-gmmlu.json in eval_tasks.py's multiple-choice format (question, options, gold letter).
Run in an image with pyarrow, --network none, on the parquet files fetched at the revision in data/SOURCES.md:

  python3 tools/g7/make_gmmlu.py <parquet dir> <out json> [--n 500 --seed 1234]
"""
import argparse
import glob
import json
import os
import random

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    pool = []
    for f in sorted(glob.glob(os.path.join(a.src, "*-test.parquet"))):
        lang = os.path.basename(f).split("-")[0]
        pool += [(lang, i, r) for i, r in enumerate(pq.read_table(f).to_pylist())]
    pick = sorted(random.Random(a.seed).sample(range(len(pool)), a.n))
    items = []
    for k in pick:
        lang, i, r = pool[k]
        items.append({"id": k, "lang": lang, "row": i, "sample_id": r["sample_id"], "question": r["question"],
                      "options": [r[f"option_{c}"] for c in "abcd"], "gold": r["answer"].strip(),
                      "category": lang, "label": r["cultural_sensitivity_label"]})
    json.dump(items, open(a.out, "w"), ensure_ascii=False, indent=0)
    langs = {}
    for it in items:
        langs[it["lang"]] = langs.get(it["lang"], 0) + 1
    print(f"{len(items)} of {len(pool)} items; languages {dict(sorted(langs.items()))}")


if __name__ == "__main__":
    main()
