#!/usr/bin/env python3
"""Freeze the task items once: GSM8K test and MMLU-Pro test, --n items each, drawn with random.Random(seed).

Same sampler as the quantised-pack-quality-gate skill's eval_data.py (and the MiMo recipe's items, n=500, seed 1234),
but read from the Hugging Face parquet files at a pinned revision instead of `datasets.load_dataset`, so the source
bytes are recorded. Needs pyarrow: run it in any image that has it (CPU only, no network):

  docker run --rm --network none --memory 4g -u $(id -u):$(id -g) -v $PWD:/w -w /w --entrypoint python3 <image> \
    tools/g7/make_items.py --gsm8k <gsm8k main/test parquet> --mmlu-pro <MMLU-Pro data/test parquet> \
    --out tools/g7/data [--n 500 --seed 1234]

Writes items-gsm8k.json and items-mmlu_pro.json (id = row index in the split, sorted).
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", required=True)
    ap.add_argument("--mmlu-pro", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = pq.read_table(a.gsm8k).to_pylist()
    pick = sorted(random.Random(a.seed).sample(range(len(g)), a.n))
    items = [{"id": i, "question": g[i]["question"],
              "gold": g[i]["answer"].split("####")[-1].strip().replace(",", "")} for i in pick]
    (out / "items-gsm8k.json").write_text(json.dumps(items, indent=0))

    m = pq.read_table(a.mmlu_pro).to_pylist()
    pick = sorted(random.Random(a.seed).sample(range(len(m)), a.n))
    items = [{"id": i, "question": m[i]["question"], "options": list(m[i]["options"]), "gold": m[i]["answer"],
              "category": m[i]["category"]} for i in pick]
    (out / "items-mmlu_pro.json").write_text(json.dumps(items, indent=0))
    print(f"GSM8K test rows {len(g)}, MMLU-Pro test rows {len(m)}; wrote {a.n} + {a.n} items (seed {a.seed}) to {out}")


if __name__ == "__main__":
    main()
