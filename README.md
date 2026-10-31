# Clover-35B-A3B-Preview on a single DGX Spark (SGLang, Docker Compose)

Serve [Clover-35B-A3B-Preview](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview) on one NVIDIA DGX Spark with SGLang. One command builds SGLang with Clover support, downloads the pinned checkpoint and starts an OpenAI-compatible API with reasoning and tool calls.

Clover is DiffuSpace's diffusion language model, built on Qwen3.6-35B-A3B: a mixture of experts with 35B parameters, about 3B active per token, and a 128K context window. It does not generate strictly one token at a time. A bidirectional pathway drafts a block of tokens at once, and the same model checks that block autoregressively and keeps the part it agrees with. On the Spark this pays off most on code, where whole stretches are predictable: one request runs at 84 tok/s on code and about 30 tok/s on prose.

**Status: proof of concept.** SGLang support is an open pull request ([sgl-project/sglang#46927](https://github.com/sgl-project/sglang/pull/46927)), pinned here by commit and patched for the Spark's GPU. Expect this setup to change once that PR lands or DiffuSpace publishes its llama.cpp code.

**Hardware:** DGX Spark (GB10, sm_121, aarch64), driver 580, CUDA 13, Docker with the NVIDIA Container Toolkit and Compose v2. About 110 GB free disk: 69 GB of BF16 weights plus a 36 GB image. The server takes about 91 GB of the Spark's memory.

## Quickstart

```bash
git clone https://github.com/dedene/Clover-35B-A3B-DGX-Spark.git
cd Clover-35B-A3B-DGX-Spark
docker compose up -d --wait     # build the image, download 69 GB (first run only), load (~4 min), wait until healthy
tools/check.py                  # model id, thinking off/on, tool call and round trip, streaming, Dutch, speed
docker compose down
```

The API is at `http://<spark-ip>:8888/v1`, model `clover`.

```bash
curl http://<spark-ip>:8888/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "clover",
  "messages": [{"role": "user", "content": "Write a Python function that merges overlapping intervals."}],
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Thinking is on by default. The reasoning comes back in `message.reasoning_content`; `"chat_template_kwargs": {"enable_thinking": false}` turns it off for a request. Tool calls use the Qwen3-Coder XML format, which SGLang parses into OpenAI `tool_calls`.

## How it is put together

- **`Dockerfile`:** the official SGLang arm64 CUDA 13 nightly of 2026-10-18, pinned by digest, with the Clover PR branch checked out over its editable install. The PR is Python and Triton only, so nothing needs compiling. Two changes on top:
  - The nightly's git config still carries the expired GitHub token of the CI run that built it, which breaks every fetch. The build removes it.
  - The PR only enables its Clover kernels on H100/H200 and B200/GB300 (compute capability 9 and 10). The Spark's GB10 is sm_121, so the build adds 12. The PR's own GPU kernel tests, which compare against reference implementations, pass on the GB10.
- **`compose.yaml`:** `download` fetches the pinned revision with the Hugging Face CLI, then `clover` serves. A healthcheck hits `/health`.
- **`tools/serve.sh`:** `sglang.launch_server` with `--speculative-algorithm CLOVER` (no separate draft model), the Qwen3 reasoning parser and the Qwen3-Coder tool parser.
- **`tools/check.py`:** functional checks and one speed line, plus `--needle N` for a long-context recall test.

## Configuration

Copy `.env.example` to `.env` and uncomment what you want to change, then `docker compose up -d --wait` again.

| Variable | Default | |
|---|---|---|
| `DRAFT_TOKENS` | 16 | Block width per step: 2, 4, 8, 16 or 32. See the speed table. |
| `ACCEPT_RULE` | `threshold` | `exact` accepts only drafts that match the verify pass's top token. |
| `CONTEXT` | 131072 | Context window, prompt and output together (native: 131072). |
| `PARALLEL` | 4 | Requests served at once. |
| `MEM_FRACTION` | 0.75 | Share of GPU memory for weights, KV cache and recurrent state. |
| `PORT`, `HOST`, `SERVED_NAME` | 8888, 0.0.0.0, clover | |
| `API_KEY` | empty | Bearer token, when set. |
| `MODEL_DIR` | `~/.cache/clover` | Where the checkpoint lives. |

## Speed

One request, thinking off, temperature 0, 600 output tokens:

| `DRAFT_TOKENS` | Code | English | Dutch |
|---|---:|---:|---:|
| 32 | 90 | 26 | 25 |
| **16 (default)** | **84** | **33** | **27** |
| 8 | 63 | 40 | 31 |
| 2 (close to one token at a time) | 28 | 29 | 29 |

On code Clover accepts about 10 drafted tokens per step. On prose it accepts 3 to 4, whatever the block width, and a wider block costs more per step (about 69 ms at width 2, 155 ms at 32). 16 has the best average.

Two caveats. The default acceptance rule is not lossless: drafts are the model's top tokens, and temperature and top_p apply only to the extra token each step adds. Different widths therefore give slightly different text. SGLang also adds the model's preferred penalties by default (presence 0.9, repetition 1.05, and two length penalties), and rejects requests with `min_p`, logprobs, logit bias or structured output.

## As an agent: benchmark results

I ran Clover through the same agent benchmark as the other models on this Spark, on 28 October 2026. Same box, same samples, temperature 0, four requests in flight, the standard [Inspect](https://inspect.aisi.org.uk/) tasks from inspect_evals 0.24.0. The server ran the defaults above (16-token blocks, threshold acceptance) for every suite.

The comparable run has thinking off and 4,096 output tokens, like every other model in the table. Clover turned out to need thinking for tool calls, so I ran the agentic suites and GPQA a second time with thinking on and 32,768 output tokens. IFEval and Dutch only ran with thinking off. Brackets are the 95% interval.

| | Clover, thinking on | Clover, thinking off | Qwen3.8 Flash Next (NVIDIA FP4-Local, vLLM + MTP) | Qwen3.8 Flash Next (MLX 4-bit, TensorFold + MTP) | Nanbeige4.3-33B (Q6_K, llama.cpp + 3B drafter) | Nemotron 3.5 Super VL (NVFP4, vLLM + LiLiCorr) |
|---|---:|---:|---:|---:|---:|---:|
| BFCL single-turn, n=500 | **89.6** (87–92) | 25.4 (22–29) | 82.4 | 71.8 | 84.6 | 85.6 |
| BFCL multi-turn, n=200 | **61.5** (55–68) | 43.5 (37–50) | 54.5 | 43.0 | 56.5 | 49.2 |
| AgentDojo, tasks done without an attack, n=96 | **86.5** (78–92) | 75.0 (65–83) | 84.4 | 78.1 | 79.2 | 77.1 |
| AgentDojo, attacks resisted, n=200 | 75.5 (69–81) | 80.0 (74–85) | **99.0** | 97.0 | 58.5 | 81.5 |
| AgentDojo, tasks done under attack, n=200 | 75.0 (69–80) | 70.5 (64–76) | **82.0** | 70.5 | 47.5 | 64.0 |
| AgentDojo, done cleanly under attack, n=200 | 61.0 (54–67) | 61.0 (54–67) | **81.0** | 69.0 | 33.0 | 55.5 |
| IFEval, prompt strict, n=541 | | 79.3 (76–82) | **85.8** | 85.0 | 74.7 | 85.6 |
| Global-MMLU Dutch, n=300 | | 80.3 (75–84) | 87.0 | **87.7** | 70.3 | 86.7 |
| GPQA-Diamond, n=198 | **78.3** (72–83) | 61.1 (54–68) | 64.6 | 59.1 | 64.6 | 56.1 |
| Agent loop, steps per minute (8 tool steps) | 22.9 | **48.5** | 13.8 | 21.0 | 5.7 | 19.2 |
| Decode, tok/s weighted over code, prose and tool calls | | 40.4 | 43.6 | **67.2** | 12.3 | 34.6 |
| Decode, Dutch prose, tok/s | | 29.4 | 31.7 | **50.2** | 9.5 | 23.6 |
| Prefill at ~32k tokens, tok/s | | **4,828** | 2,274 | 2,185 | 469 | 2,373 |
| 4 requests at once, tok/s | | 104 | 113 | **145** | 23.6 | 93 |
| Memory for the whole server | ~96 GB | ~96 GB | ~97 GB | ~103 GB | ~98 GB | **~91 GB** |

The speed rows don't depend on thinking, so they appear once. Done cleanly means the task got done and the injected instruction did not.

Without thinking, Clover hardly calls tools. Asked a BFCL question with the right tool at hand, it usually explains how one would get the answer instead: simple 4%, multiple 8%, parallel 6%, parallel-multiple 10%, while irrelevance (don't call a tool when none fits) is 99%. It isn't the drafting. With `ACCEPT_RULE=exact`, which matches plain greedy decoding, the answers are the same prose.

With thinking on it is the best tool user I have measured on one Spark: 89.6 on BFCL single-turn, 61.5 on multi-turn and 86.5 on AgentDojo tasks, all above Qwen FP4-Local. It also becomes a little eager, with irrelevance dropping from 99 to 88. GPQA-Diamond goes from 61.1 to 78.3. DiffuSpace reports 83.8; this run is temperature 0 and capped at 32,768 tokens.

Prompt injection is still the weak spot. With thinking, 49 of 200 injected instructions got through, 40 without, against 2 for Qwen FP4-Local. Under attack it finishes 61% of the tasks cleanly in both modes, against 81% for Qwen.

Speed is mixed, and the reason is the block drafting. Code and tool calls come out at 64-67 tok/s, prose at 28-29, a bit below Qwen FP4-Local. Prefill is twice as fast as anything else here. In the agent loop, where replies are short and the prompt grows to 29K tokens, Clover does 48.5 steps per minute without thinking and 22.9 with it, against 13.8 for Qwen FP4-Local.

IFEval (79.3) and Dutch (80.3) sit 6-7 points below Qwen. Clover thinks the capital of West Flanders is Gent.

So Qwen FP4-Local stays the model for an agent on a single Spark, for one reason: an agent that reads mail, chat or web pages has to ignore what's written in them, and Qwen resists 99% of the attacks to Clover's 75%. On everything else Clover with thinking on matches or beats it. It is also a preview on an unmerged SGLang branch. If the next version resists injections better, this changes.

## License

The scripts in this repository are MIT. Clover-35B-A3B-Preview is Apache-2.0, derived from Qwen3.6-35B-A3B.
