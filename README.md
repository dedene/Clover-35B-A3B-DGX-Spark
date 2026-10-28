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

## License

The scripts in this repository are MIT. Clover-35B-A3B-Preview is Apache-2.0, derived from Qwen3.6-35B-A3B.
