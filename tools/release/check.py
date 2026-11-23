#!/usr/bin/env python3
"""Release check of a served Inkling 2 Mini (stdlib only): the replies an agent client sees.

  python3 tools/release/check.py URL OUT.json [PICTURE...]

At the vendor's sampling (temperature 1.0, no cut) with a fixed seed and low reasoning effort:
  chat      one chat whole and the same request streamed: both finish with stop and the same content
  tools     a shell tool, turn 1 whole and streamed (same call, no tool name or markup in the streamed text),
            then turn 2 with the tool's answer: a new call whose id differs from turn 1's
  image     one chat per picture (JPEG, WebP): the reply finishes with stop and says something
Prints one PASS/FAIL line per check and RELEASE_CHECK=PASS|FAIL; writes every request and reply to OUT.json.
"""
import base64
import json
import mimetypes
import sys
import urllib.request

URL = sys.argv[1].rstrip("/")
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0, "seed": 7, "reasoning_effort": "low",
            "max_tokens": 4096}
TOOLS = [{"type": "function", "function": {
    "name": "run_shell_command", "description": "Run a shell command and return its output.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
ASK = [{"role": "user", "content": "List the files in /tmp using the shell tool."}]
log: list = []
results: list = []


def model() -> str:
    with urllib.request.urlopen(URL + "/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def post(body: dict):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900)


def whole(body: dict) -> tuple[dict, str]:
    j = json.load(post(body))
    log.append({"request": body, "response": j})
    c = j["choices"][0]
    return c["message"], c["finish_reason"]


def streamed(body: dict) -> tuple[str, str, list, str]:
    body = dict(body, stream=True)
    content, reasoning, calls, finish = "", "", {}, None
    for line in post(body):
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ch = json.loads(line[6:]).get("choices") or []
        if not ch:
            continue
        d, finish = ch[0].get("delta", {}), ch[0].get("finish_reason") or finish
        content += d.get("content") or ""
        reasoning += d.get("reasoning_content") or ""
        for tc in d.get("tool_calls") or []:
            c = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
            c["id"] = tc.get("id") or c["id"]
            c["name"] += (tc.get("function") or {}).get("name") or ""
            c["arguments"] += (tc.get("function") or {}).get("arguments") or ""
    log.append({"request": body, "streamed": {"content": content, "reasoning": reasoning,
                                              "calls": list(calls.values()), "finish": finish}})
    return content, reasoning, list(calls.values()), finish


def check(name: str, ok: bool, detail: str) -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)


def main() -> None:
    m = model()
    base = {"model": m, **SAMPLING}
    chat = dict(base, messages=[{"role": "user", "content": "Name three uses of a hash map, one line each."}])
    msg, fin = whole(chat)
    check("chat whole", fin == "stop" and bool(msg.get("content")), f"finish={fin} content={msg.get('content')!r:.160}")
    content, _, _, sfin = streamed(chat)
    check("chat streamed", sfin == "stop" and content == msg.get("content"),
          f"finish={sfin}, same content as whole: {content == msg.get('content')}")

    t1 = dict(base, messages=ASK, tools=TOOLS)
    msg, fin = whole(t1)
    calls = msg.get("tool_calls") or []
    check("tools turn 1 whole", fin == "tool_calls" and len(calls) == 1 and
          calls[0]["function"]["name"] == "run_shell_command",
          f"finish={fin} calls={json.dumps(calls)[:200]} content={msg.get('content')!r:.80}")
    content, _, scalls, sfin = streamed(t1)
    same = bool(calls) and len(scalls) == 1 and scalls[0]["name"] == calls[0]["function"]["name"] and \
        scalls[0]["arguments"] == calls[0]["function"]["arguments"]
    clean = "run_shell_command" not in content and "<|" not in content
    check("tools turn 1 streamed", sfin == "tool_calls" and same and clean,
          f"finish={sfin} same call as whole: {same}, text free of tool markup: {clean} ({content!r:.80})")
    if calls:
        msgs = ASK + [{"role": "assistant", "content": "", "tool_calls": calls[:1]},
                      {"role": "tool", "tool_call_id": calls[0]["id"], "content": "a.txt\nb.log"},
                      {"role": "user", "content": "Now show the first line of /tmp/a.txt with the shell tool."}]
        msg2, fin2 = whole(dict(base, messages=msgs, tools=TOOLS))
        calls2 = msg2.get("tool_calls") or []
        new_id = bool(calls2) and calls2[0].get("id") != calls[0].get("id")
        check("tools turn 2", fin2 == "tool_calls" and new_id,
              f"finish={fin2} calls={json.dumps(calls2)[:200]} id differs from turn 1: {new_id}")
        content2, _, scalls2, sfin2 = streamed(dict(base, messages=msgs, tools=TOOLS))
        sid = bool(scalls2) and scalls2[0]["id"] != calls[0].get("id")
        check("tools turn 2 streamed", sfin2 == "tool_calls" and sid and "run_shell_command" not in content2,
              f"finish={sfin2} id differs from turn 1: {sid}")

    for pic in sys.argv[3:]:
        mime = mimetypes.guess_type(pic)[0] or "image/jpeg"
        data = base64.b64encode(open(pic, "rb").read()).decode()
        body = dict(base, messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}},
            {"type": "text", "text": "What is in this picture? Answer in two sentences."}]}])
        msg, fin = whole(body)
        log[-1]["request"]["messages"][0]["content"][0]["image_url"]["url"] = f"<{pic}>"
        check(f"image {pic.rsplit('/', 1)[-1]}", fin == "stop" and bool(msg.get("content")),
              f"finish={fin} content={msg.get('content')!r:.200}")

    json.dump(log, open(sys.argv[2], "w"), indent=1)
    print(f"RELEASE_CHECK={'PASS' if all(results) else 'FAIL'} ({sum(results)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
