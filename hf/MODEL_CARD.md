---
license: apache-2.0
base_model: thinkingmachines/Inkling-2-Mini
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: exllamav3
tags:
- exl3
- inkling2
- tensorfold
- dgx-spark
- gb10
---

# Inkling-2-Mini, EXL3 at ~4 bpw

[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (26.4B mixture of experts,
3.4B active, a looped middle) quantised to EXL3 from its NVFP4/MXFP8 release, made to run across two NVIDIA DGX Sparks
(GB10) with tensor parallelism at the model's full 1M-token context, with its vision tower and its DSpark drafter.
The project that serves it: [ajensenwaud/inkspark-mini-2dgx](https://github.com/ajensenwaud/inkspark-mini-2dgx).

| | |
|---|---|
| Weights | 15.6 GB: 3 shards (14.3 GB), drafter 0.64 GB, vision tower 0.64 GB; about 11.6 GiB a Spark at TP=2 |
| Bitrate | experts 4 bpw (gate, up, down); attention 6 bpw; head 8 bpw; 4.05 bpw over the layers |
| Kept as released | router, norms, convolutions, the loop's gates and messages, embedding (bf16) |
| Conversion | exllamav3 1.6.6 with an Inkling 2 Mini architecture overlay (all three passes of the looped layers in each matrix's calibration), calibration 250 x 2048, mul1 codebook; source @ `ed40e7da` |
| Drafter | the model's own DSpark drafter (`mtp.safetensors`), bf16 as released |
| Vision | the model's vision tower (`vision.safetensors`), bf16 as released |

Against the release itself (its NVFP4/MXFP8 experts as stored), streamed through exllamav3 on the same rows:

| Text (rows of 2,048 tokens) | Mean KL | Same top token | Perplexity, release / this pack |
|---|---|---|---|
| wikitext-2 (64) | 0.0069 | 96.5 % | 6.4448 / 6.4536 (+0.14 %) |
| code (16) | 0.0058 | 98.1 % | 2.0634 / 2.0649 (+0.07 %) |

## Quality

Served on two Sparks with InkSpark-Mini's engine, fp8 KV cache, temperature 1.0 (nothing cut), reasoning effort high,
64K tokens a reply.

| Task | This pack | Thinking Machines' card |
|---|---|---|
| GSM8K (500) | 97.0 % | not reported |
| Global-MMLU-Lite (500 items, 23 languages) | 82.2 % | not reported |
| MMLU-Pro (500) | 81.4 % | not reported |

The card reports none of these tasks, so they measure the pack, not its distance from the release; that distance is
the KL table above.

## Speed (two DGX Sparks)

| | |
|---|---|
| One stream, short prompt, greedy (chat / code / agent) | 73 / 95 / 78 tok/s with the drafter |
| One stream at temperature 1.0 (chat / code / agent) | 70 / 83 / 80 tok/s |
| Two streams at once, greedy | 111 tok/s in total with the drafter |
| Eight streams at once (8 x 128K windows) | 242 tok/s in total |
| Decode at 128K / 512K / 1M context | 59.0 / 25.1 / 16.7 tok/s |
| Reading a cold prompt, 120K / 950K tokens | 1,646 tok/s (73 s) / 981 tok/s (16 min) |

Follow-up turns resume from the prompt cache in seconds. Performance varies for cold long prompts.

## Use

With InkSpark-Mini on two Sparks: `scripts/prepare.sh`, `scripts/fetch_pack.sh`, then `./start.sh`. The pack needs
InkSpark-Mini's TensorFold patches (the `inkling2` family with Mini's loop); upstream TensorFold and exllamav3 cannot
load it yet. The model reasons before it answers even at the lowest effort, so give replies thousands of tokens.

## Acknowledgements

Thinking Machines for the model, its vision tower and the DSpark drafter; turboderp for exllamav3, which made this
pack; Ash Hart for TensorFold, the engine it was evaluated on. Evaluation data: GSM8K (MIT), MMLU-Pro (MIT),
Global-MMLU-Lite (Apache-2.0).

## Licence

Apache-2.0, as the source model.
