# SGLang with Clover support: the official arm64 CUDA 13 nightly plus the Clover PR branch (sgl-project/sglang#46927).
# The PR branches off main of 2026-10-16 (7c99ace); this nightly is 71 commits later with the same dependency pins
# (flashinfer 0.7.2). The PR is Python and Triton only, so checking it out over the editable install is the whole
# change: no extension to compile.
ARG BASE=lmsysorg/sglang:nightly-dev-cu13-20261018-b234b5c3@sha256:bfef46e5a45a815f8c2f88a722980baf47b71a8689b89c6cc738f596ceeaf47c
FROM ${BASE}

ARG SGLANG_REPO=https://github.com/WilliamZR/sglang
ARG SGLANG_COMMIT=058939af24d5cdc44bb4b8852d0ddfab555ed71e
# Clover's preview kernels (GDN attention, short conv, MoE block coreset) are plain Triton but gated to compute
# capability 9 and 10 (H100/H200, B200/GB300). The GB10 is sm_121: open the gates to major 12 as well, in the
# kernels and in the coreset test. The PR's GPU kernel tests check the result against reference implementations.
ARG CLOVER_GATES="python/sglang/kernels/ops/attention/clover_gdn_triton.py python/sglang/kernels/ops/mamba/clover_conv_triton.py python/sglang/kernels/ops/moe/clover_coreset_triton.py test/registered/kernels/ops/moe/test_clover_coreset.py"
# Clover rejects any constrained decoding (json_schema, regex, structural_tag). SGLang turns `"strict": true` on a
# tool with tool_choice "auto" into a structural tag, so such requests fail with 400 ("CLOVER does not support:
# structural_tag"); OpenAI clients such as Inspect set strict on every tool. Ignore strict in auto mode: the
# Qwen3-Coder parser still turns the output into tool_calls, as llama.cpp does. "required" and named tool_choice
# stay constrained, and Clover still rejects those.
ARG STRICT_SITE=python/sglang/srt/function_call/function_call_parser.py
# The nightly's .git/config keeps the expired auth header of the CI run that built it; git sends it to github.com
# and every fetch fails ("could not read Username"). Drop it first.
RUN cd /sgl-workspace/sglang \
 && git config --unset-all http.https://github.com/.extraheader \
 && git fetch --depth 1 "${SGLANG_REPO}" "${SGLANG_COMMIT}" \
 && git -c advice.detachedHead=false checkout --detach FETCH_HEAD \
 && test "$(git rev-parse HEAD)" = "${SGLANG_COMMIT}" \
 && sed -i 's/get_device_capability(\(.*\))\[0\] in (9, 10)/get_device_capability(\1)[0] in (9, 10, 12)/' ${CLOVER_GATES} \
 && test "$(grep -l 'in (9, 10, 12)' ${CLOVER_GATES} | wc -l)" = 4 \
 && sed -i 's/^\( *\)any(tool\.function\.strict for tool in self\.tools)$/\1False  # Clover: tool strict mode is not a constraint/' ${STRICT_SITE} \
 && grep -q 'False  # Clover: tool strict mode' ${STRICT_SITE} \
 && python3 -c "import sglang.srt.models.clover_moe, sglang.srt.speculative.clover_worker"
