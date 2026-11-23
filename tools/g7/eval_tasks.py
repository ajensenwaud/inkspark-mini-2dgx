#!/usr/bin/env python3
"""G7 task accuracy through the served API (stdlib only): GSM8K, MMLU-Pro and Global-MMLU-Lite on the frozen items in
tools/g7/data.

InkSpark's tools/g8/eval_tasks.py (itself from LingSpark tools/eval/eval_tasks.py), defaults changed for Mini.
Inkling 2 always reasons; its template takes a reasoning_effort (default 0.9, "high") instead of enable_thinking, and
--effort (default high, the template's own) is sent as the request's reasoning_effort. Sampling is sent explicitly
in every request: temperature 1.0 (Mini's generation_config.json), top_p 1.0, top_k 0 (off), min_p 0 and a fixed
seed. One generous token budget (reasoning counts against max_tokens; a capped reply scores as wrong and measures the
cap, not the model).

  python3 tools/g7/eval_tasks.py http://127.0.0.1:8888 Inkling-2-Mini results/g7/evals --tag fp8 \
      [--tasks gsm8k,mmlu_pro,gmmlu] [--concurrency 8] [--max-tokens 65536] [--limit N] [--ids 3,5] [--effort high]

Writes <dir>/<tag>-<task>.jsonl: one line per item with the parsed answer, finish reason, token counts, the request
body and the server's raw JSON response (reasoning_content included). <tag>-<task>.score.json: accuracy, 95%
interval, unparsed / truncated / error counts, generated tokens, by category, settings. Resumes: items already in
the .jsonl are not asked again; every scoring pass re-parses the saved replies with the current parser.
Truncation guard: after the first --check-after replies, more than 2% finished by length stops the run (exit 3).
"""

import argparse
import concurrent.futures as cf
import json
import math
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

LETTERS = "ABCDEFGHIJ"
DATA = Path(__file__).resolve().parent / "data"


def prompt(task: str, it: dict) -> str:
    if task == "gsm8k":
        return (f"{it['question']}\n\nSolve the problem step by step. On the last line, write the final answer as "
                f"'Answer: <number>'.")
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(it["options"]))
    return (f"{it['question']}\n\nOptions:\n{opts}\n\nThink step by step, then on the last line write the final "
            f"answer as 'Answer: <letter>'.")


def answer_text(text: str) -> str:
    """The part of a reply after the reasoning: servers that leave <think>...</think> in content are handled too."""
    return text.split("</think>")[-1] if "</think>" in text else text


def parse(task: str, text: str) -> str | None:
    text = answer_text(text or "")
    if task == "gsm8k":
        m = re.findall(r"Answer:\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", text)
        nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
        if not nums:
            return None
        v = nums[-1].replace(",", "").rstrip(".")
        try:
            f = float(v)
        except ValueError:
            return None
        return str(int(f)) if f == int(f) else str(f)
    m = re.findall(r"Answer:\s*\**\s*\(?([A-J])\)?", text)
    if not m:
        m = re.findall(r"\b(?:answer is|answer:)\s*\**\s*\(?([A-J])\)?", text, re.I)
    if not m:                                              # e.g. "This corresponds to option **F**."
        m = re.findall(r"\boption\s*\**\(?([A-J])\)?\**", text, re.I)
    return m[-1].upper() if m else None


def correct(task: str, got: str | None, gold: str) -> bool:
    if got is None:
        return False
    return float(got) == float(gold) if task == "gsm8k" else got == gold


def body_for(a, model: str, task: str, it: dict) -> dict:
    b = {"model": model, "messages": [{"role": "user", "content": prompt(task, it)}], "max_tokens": a.max_tokens,
         "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "min_p": a.min_p, "seed": a.seed}
    if a.effort:
        b["reasoning_effort"] = a.effort
    return b


def ask(a, url: str, model: str, task: str, it: dict) -> dict:
    body = body_for(a, model, task, it)
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    j: dict = {}
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=a.timeout) as r:
                j = json.load(r)
            break
        except Exception as exc:                          # noqa: BLE001  (retried, then recorded)
            if attempt == 2:
                return {"id": it["id"], "error": str(exc), "request": body}
            time.sleep(5)
    msg = j["choices"][0]["message"]
    text = msg.get("content") or ""
    got = parse(task, text)
    usage = j.get("usage") or {}
    return {"id": it["id"], "got": got, "gold": it["gold"], "correct": correct(task, got, it["gold"]),
            "finish": j["choices"][0].get("finish_reason"), "tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"), "category": it.get("category"), "text": text,
            "reasoning_chars": len(msg.get("reasoning_content") or ""), "wall_s": round(time.time() - t0, 1),
            "request": body, "response": j}


def load_done(raw: Path, task: str) -> dict:
    done = {}
    if raw.exists():
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                r["got"] = parse(task, r["text"])             # scores always use the current parser
                r["correct"] = correct(task, r["got"], r["gold"])
                done[r["id"]] = r
    return done


def get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return json.load(r)
    except Exception:                                      # noqa: BLE001
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("url")
    ap.add_argument("model")
    ap.add_argument("dir", help="output directory (results/g7/evals)")
    ap.add_argument("--tag", required=True, help="arm name, e.g. pack")
    ap.add_argument("--tasks", default="gsm8k,mmlu_pro,gmmlu")
    ap.add_argument("--items", default=str(DATA), help="directory holding items-<task>.json (default tools/g7/data)")
    ap.add_argument("--concurrency", type=int, default=8, help="match the serve's PARALLEL")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="", help="comma-separated item ids to ask (default: all)")
    ap.add_argument("--max-tokens", type=int, default=65536,
                    help="per-request budget incl. reasoning (default 64K, half a 128K slot: a reply that long is a loop)")
    ap.add_argument("--effort", default="high", help="reasoning_effort sent to the template ('' sends none)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timeout", type=int, default=7200, help="per-request seconds")
    ap.add_argument("--check-after", type=int, default=0, help="truncation check after this many replies (0: off)")
    ap.add_argument("--serve-config", default="", help="file to record with the score (start.sh's logs/serve-config.txt)")
    a = ap.parse_args()
    url = a.url.rstrip("/")
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k, v in body_for(a, a.model, "gsm8k", {"question": "", "id": 0}).items()
                if k not in ("model", "messages")}
    server = {"models": get_json(url + "/v1/models"), "health": get_json(url + "/health")}
    serve_cfg = Path(a.serve_config).read_text() if a.serve_config and Path(a.serve_config).exists() else None
    for task in a.tasks.split(","):
        items = json.loads((Path(a.items) / f"items-{task}.json").read_text())
        if a.ids:
            want = {int(x) for x in a.ids.split(",")}
            items = [it for it in items if it["id"] in want]
        if a.limit:
            items = items[:a.limit]
        raw = d / f"{a.tag}-{task}.jsonl"
        done = load_done(raw, task)
        todo = [it for it in items if it["id"] not in done]
        print(f"[{time.strftime('%H:%M:%S')}] {a.tag} {task}: {len(done)} done, {len(todo)} to ask", flush=True)
        t0 = time.time()
        stop = threading.Event()
        new, trunc, errors = 0, 0, 0
        with raw.open("a") as f, cf.ThreadPoolExecutor(a.concurrency) as ex:
            futs = [ex.submit(lambda it=it: None if stop.is_set() else ask(a, url, a.model, task, it)) for it in todo]
            for fu in cf.as_completed(futs):
                r = None if fu.cancelled() else fu.result()
                if r is None:
                    continue
                f.write(json.dumps(r) + "\n")
                f.flush()
                if "error" in r:
                    errors += 1
                    continue
                done[r["id"]] = r
                new += 1
                trunc += r["finish"] == "length"
                if new % 10 == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] {task} {len(done)}/{len(items)}  truncated {trunc}"
                          f"  errors {errors}  last {r['tokens']} tok", flush=True)
                if a.check_after and new == a.check_after and trunc > 0.02 * new and not stop.is_set():
                    stop.set()
                    for x in futs:
                        x.cancel()
                    print(f"STOP: {trunc}/{new} replies truncated at max_tokens {a.max_tokens} (> 2%): raise the "
                          f"budget, move {raw} to {d}/discarded/ and restart", flush=True)
        if stop.is_set():
            sys.exit(3)
        rs = [done[it["id"]] for it in items if it["id"] in done]
        n, c = len(rs), sum(r["correct"] for r in rs)
        acc = c / n if n else 0.0
        half = 1.96 * math.sqrt(acc * (1 - acc) / n) if n else 0.0
        cats: dict = {}
        for r in rs:
            if r.get("category"):
                cats.setdefault(r["category"], [0, 0])
                cats[r["category"]][0] += r["correct"]
                cats[r["category"]][1] += 1
        score = {"tag": a.tag, "task": task, "n": n, "of": len(items), "correct": c, "acc": round(acc, 4),
                 "ci95": round(half, 4), "unparsed": sum(r["got"] is None for r in rs),
                 "truncated": sum(r["finish"] == "length" for r in rs), "errors_this_pass": errors,
                 "gen_tokens": sum(r["tokens"] or 0 for r in rs),
                 "mean_tokens": round(sum(r["tokens"] or 0 for r in rs) / n, 1) if n else None,
                 "wall_s_this_pass": round(time.time() - t0, 1), "by_category": cats or None,
                 "settings": {**settings, "concurrency": a.concurrency}, "server": server,
                 "serve_config": serve_cfg}
        (d / f"{a.tag}-{task}.score.json").write_text(json.dumps(score, indent=1))
        print(json.dumps({k: v for k, v in score.items() if k not in ("by_category", "server", "serve_config")}),
              flush=True)


if __name__ == "__main__":
    main()
