# InkSpark-Mini: Inkling-2-Mini on 2x DGX Spark (GB10) with TensorFold

Thinking Machines' Inkling-2-Mini (26.4B parameters, 3.4B active; MoE loop architecture)
served across two DGX Sparks with tensor parallelism, from our own EXL3 quantisation of Inkling's
NVFP4/MXFP8 release, on TensorFold v1.3.3. The recipe uses the model's own DSpark drafter.

## Running it

What you need:
- Two GB10 DGX Sparks joined by their CX7 ports
- Docker with the NVIDIA runtime on both
- SSH configured from the head to the worker.

Ports: 8888 (API), 29592 (between the Sparks).

Copy `scripts/local.sh.example` to `scripts/local.sh` and set `WORKER` (user@address) and `MASTER` (the head's CX7
address).

Commands:
```bash
scripts/prepare.sh        # the image, then TensorFold v1.3.3 + patches-zig/ compiled in it, on both Sparks
scripts/fetch_pack.sh     # the model from Hugging Face (pinned, 15.6 GB), on both Sparks, verified
./start.sh                # OpenAI API on :8888, model id Inkling-2-Mini
./start.sh restart
./stop.sh
```

The model: [ajensenwaud/inkspark-mini-2dgx](https://huggingface.co/ajensenwaud/inkspark-mini-2dgx).

Settings in `scripts/config.sh` (override in `scripts/local.sh` or on the command line): `SHAPE=long` (one
conversation at 1M, the default) or `SHAPE=agents` (eight at 128K each), `DRAFTER=1|0`, `KV=fp8|bf16`, `VISION=1|0`,
`PORT`, `HOST`.

## Model

| | |
|---|---|
| Size | 15.6 GB, about 11.6 GiB on each Spark |
| Experts | 4 bpw |
| Attention, head | 6 bpw, 8 bpw |
| Kept as released | router, norms, convolutions, the loop's gates and messages, embedding (bf16) |
| Drafter, vision | bf16 as released by Thinking Machines; the drafter's projections are rounded to NVFP4 when loaded (`DRAFT_WEIGHTS=bf16` keeps them) |

## Quality

Settings: fp8 KV, temperature 1.0, reasoning high, 64K tokens / reply:

| Task | InkSpark-Mini | Thinking Machines' card |
|---|---|---|
| GSM8K (500) | 97.0 % | not reported |
| Global-MMLU-Lite (500 questions, 23 languages) | 82.2 % | not reported |
| MMLU-Pro (500) | 81.4 % | not reported |

## Speed

| Streams | Greedy | Thinking Machines' sampling (temperature 1.0) |
|---|---|---|
| 1 (chat / code / agent / report) | 77 / 89 / 70 / 69 tok/s | 68 / 94 / 79 / 66 tok/s |
| 2 | 116 tok/s in total | 108 tok/s in total |
| 3 | 137 tok/s in total | 137 tok/s in total |
| 4 | 153 tok/s in total | 156 tok/s in total |
| 8 (`SHAPE=agents`) | 232 tok/s in total | 233 tok/s in total |


Long prompts, one stream, greedy:

| Context | Reading the prompt | First token | Decode (drafter on) |
|---|---|---|---|
| 16K | 1,827 tok/s | 9 s | 66.5 tok/s |
| 128K | 1,638 tok/s | 80 s | 59.0 tok/s |
| 256K | 1,432 tok/s | 3.0 min | 36.5 tok/s |
| 512K | 1,254 tok/s | 6.9 min | 25.1 tok/s |
| 1M | 899 tok/s | 19.2 min | 16.7 tok/s |

## Credits and licence

Thanks to Thinking Machines for Inkling-2-Mini, its vision tower and its DSpark drafter; Ash Hart
([@ashxhart](https://x.com/ashxhart)) for [TensorFold](https://github.com/ashhart/TensorFold); turboderp for
[exllamav3](https://github.com/turboderp-org/exllamav3), which made the pack; the transformers, vLLM and SGLang
contributors whose Inkling 2 Mini code was the reference for the port; Mia AI Lab for the TensorFold CUDA base image.
Evaluation data: GSM8K (MIT), MMLU-Pro (MIT), Global-MMLU-Lite (Apache-2.0). Full list in `CREDITS.md`.

Recipe code: BSD 3-Clause. `patches-zig/` keep TensorFold's Apache-2.0. The model is Apache-2.0.
