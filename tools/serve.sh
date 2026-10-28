#!/usr/bin/env bash
# sglang.launch_server arguments for Clover-35B-A3B-Preview. Clover drafts a block of tokens with its own
# bidirectional pathway and verifies it autoregressively in the same model (--speculative-algorithm CLOVER, no
# separate drafter). Extra arguments from compose are appended.
set -euo pipefail

# Batch sizes to capture CUDA graphs for, space separated.
read -ra graph_bs <<<"${CUDA_GRAPH_BS:-1 2 4}"
args=(
  --model-path /models/Clover-35B-A3B-Preview
  --served-model-name "${SERVED_NAME:-clover}"
  --host "${HOST:-0.0.0.0}"
  --port "${PORT:-8888}"
  --trust-remote-code
  --speculative-algorithm CLOVER
  --speculative-num-draft-tokens "${DRAFT_TOKENS:-16}"
  --speculative-clover-accept-rule "${ACCEPT_RULE:-threshold}"
  --context-length "${CONTEXT:-131072}"
  --max-running-requests "${PARALLEL:-4}"
  --mem-fraction-static "${MEM_FRACTION:-0.75}"
  --cuda-graph-bs-decode "${graph_bs[@]}"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
)
[[ -n ${API_KEY:-} ]] && args+=(--api-key "${API_KEY}")

exec python3 -m sglang.launch_server "${args[@]}" "$@"
