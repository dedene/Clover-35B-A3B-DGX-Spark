# AGENTS.md

Serve Clover-35B-A3B-Preview (DiffuSpace's block-diffusion MoE, derived from Qwen3.6-35B-A3B, BF16 69 GB) on one
DGX Spark with SGLang via Docker Compose. Proof of concept: SGLang support is an open PR.

## Layout
- `Dockerfile`: the official SGLang arm64 CUDA 13 nightly plus the Clover PR branch checked out over its editable
  install, with the kernel capability gates opened to sm_12x. Python and Triton only, so nothing compiles.
- `compose.yaml`: `download` (hf CLI of the pinned revision into `MODEL_DIR`) -> `clover` (`sglang.launch_server`,
  healthcheck on `/health`). Knobs are `${VAR:-default}` plus `tools/serve.sh`. `.env.example` documents them.
- `tools/download.sh`, `tools/serve.sh`: fetch and launch. `tools/check.py`: functional checks plus one speed line
  (+ `--needle N`). The checkpoint directory name is repeated in both scripts.

## Spark
- Same box as `~/Development/Underdog-Saluki-27B-DGX-Spark` (`ssh zenjoy@10.0.0.158`). Checkout at
  `~/Development/Clover-35B-A3B-DGX-Spark`. Checkpoint in `~/.cache/clover`.
- GB10, sm_121, aarch64, 128 GB unified (121.6 GiB visible), driver 580, CUDA 13, Compose v5.
- Port 8888 is Saluki's for Hermes. Stop Saluki for Clover, start it again afterwards.

## Verify
- `docker compose config --quiet`, `shellcheck tools/*.sh`, `python3 -m py_compile tools/*.py`
- `docker compose up -d --wait`, then `tools/check.py` (all ok).
- Kernel numerics on the GB10: the PR's own GPU tests, inside the image:
  `docker run --rm --gpus all --ipc host --entrypoint bash clover-sglang:058939af -c "cd /sgl-workspace/sglang &&
  python3 -m pytest -q test/registered/kernels/ops/attention/test_clover_gdn.py
  test/registered/kernels/ops/mamba/test_clover_conv.py test/registered/kernels/ops/mamba/test_clover_verify_conv.py
  test/registered/kernels/ops/moe/test_clover_coreset.py test/registered/kernels/ops/sampling/test_clover_penalty.py"`

## Pins
- Base `lmsysorg/sglang:nightly-dev-cu13-20261018-b234b5c3@sha256:bfef46e5…` (torch 2.14.1+cu130, triton 3.8.0,
  flashinfer 0.7.2). The PR branches off main at `7c99ace` (2026-10-16); this nightly is 71 commits later with
  the same dependency pins. The 10-16 nightly predates the flashinfer 0.7.2 bump.
- SGLang PR sgl-project/sglang#46927, `WilliamZR/sglang@058939af` (feature/clover-support, 3 commits).
- Weights `Diffuspace/Clover-35B-A3B-Preview@4b0402b65e523d0183a14c431955438714cb16ff`.

## Model facts
- 40 layers: 30 Gated DeltaNet + 10 full attention (2 KV heads, head dim 256), 256 experts top-8 + 1 shared,
  hidden 2048, 128K context. Text only. Chat template is Qwen3.6's: `<think>` opened by the generation prompt,
  `chat_template_kwargs.enable_thinking: false` closes it; tool calls in the Qwen3-Coder XML format
  (`--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`).
- Serving is `--speculative-algorithm CLOVER`: the bidirectional pathway drafts a block, the autoregressive pathway
  verifies it in the same model. The default threshold acceptance rule is not distribution-preserving: drafts are
  argmax, and temperature/top_p apply to the bonus token only. `exact` accepts only argmax matches.
- SGLang adds `presence_penalty 0.9`, `repetition_penalty 1.05` and the dLLM length/think-end penalties by default.
  Requests with `min_p`, logprobs, logit bias or structured output (json_schema, regex) are rejected.

## Findings (2026-10-28, bring-up)
- The nightly's `.git/config` holds the expired auth header of the CI run that built it; every `git fetch` from
  github.com fails until it is unset.
- The PR gates its GDN, conv and coreset Triton kernels to compute capability 9 and 10; on sm_121 the server
  dies in CUDA graph capture ("BD-GDN production preview has no compatible convolution kernel"). With major 12
  added the PR's GPU kernel tests pass on the GB10 (16 tests, 46 subtests).
- Load: weights 178-197 s, 63.7-66 GB. `MEM_FRACTION 0.75` leaves 539K tokens of KV.
- `tools/check.py` all ok: thinking off/on, tool call, round trip, streaming. West Flanders' capital: Gent (should
  be Brugge).
- Single-stream decode, thinking off, temperature 0, 600 tokens (code / English / Dutch tok/s, accept length):
  DRAFT_TOKENS 32: 90/26/25 (~10 on code, ~4 on prose); 16: 84/33/27; 8: 63/40/31; 2 (near-AR): 28/29/29.
  A step costs ~69 ms at width 2 and ~155 ms at 32; prose accepts 3-4 tokens whatever the width, so wide blocks
  only pay off on code. Default 16 (best mean). Different widths give different greedy text: the threshold rule
  is not lossless.
- Startup sets `max_running_requests` to 3 at width 32 and 4 at 16 (mamba state slots).
- Over ssh, a heredoc that runs `ssh` without `-n` feeds the rest of the script to the remote shell; later local
  commands silently never run.
