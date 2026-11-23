# Credits

InkSpark-Mini stands on other people's work. Thank you to:

- **Thinking Machines** for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (Apache-2.0),
  the model InkSpark-Mini serves, with its vision tower and its bundled DSpark drafter.
- **TensorFold** ([ashhart/TensorFold](https://github.com/ashhart/TensorFold), Apache-2.0) by Ash Hart. The engine,
  its server, kernels and build are TensorFold's. InkSpark-Mini is a set of patches on top of release v1.3.3
  (`patches-zig/`), and those patches stay under Apache-2.0.
- **turboderp** for [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT): the EXL3 format and quantiser used
  to build the pack, and the reference forward the engine is checked against.
- **Hugging Face transformers** ([huggingface/transformers](https://github.com/huggingface/transformers), Apache-2.0):
  its `inkling2_loop` model code is the architecture reference for the loop.
- **vLLM** ([vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0) and **SGLang**
  ([sgl-project/sglang](https://github.com/sgl-project/sglang), Apache-2.0): their Inkling 2 Mini code is the second
  reference for the loop and the drafter's taps.
- **Mia's AI Lab** ([MiaAI-Lab](https://github.com/MiaAI-Lab)) for the GLM-5.3 TensorFold image used as the CUDA base of
  `docker/Dockerfile.build`.
- **Zig** ([ziglang.org](https://ziglang.org), MIT): the compiler TensorFold's engine is built with.
- **libjpeg-turbo** ([libjpeg-turbo.org](https://libjpeg-turbo.org), IJG/BSD-style) and **libwebp**
  ([chromium.googlesource.com/webm/libwebp](https://chromium.googlesource.com/webm/libwebp), BSD-3-Clause): built into
  the image so the server decodes JPEG and WebP pictures.
- Evaluation data in `tools/g7/data`: **GSM8K** (Cobbe et al., MIT), **MMLU-Pro** (TIGER-Lab, MIT) and
  **Global-MMLU-Lite** (Cohere Labs, Apache-2.0).

Any errors in InkSpark-Mini are mine, not theirs.
