# Trie-head beam benchmarks

This directory contains the scripts used to validate the trie-conditioned SID
output head, native shared-context attention, and runtime mixed beam widths.
They cover end-to-end throughput, ordered correctness, GPU memory, profiling,
CUDA-graph smoke tests, and the kernel experiments that informed the
implementation.

Model bundles, prompt corpora, valid-SID indexes, generated profiles, and raw
benchmark outputs are intentionally not checked in. Supply those paths at run
time.

## Safety

Inspect `nvidia-smi` first and select a genuinely free GPU. Every GPU benchmark
requires exactly one visible device:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not blanket-kill GPU or Python processes. If a benchmark server must be
stopped, terminate only the exact PID that you started.

## Scripts

| Script | Purpose |
|---|---|
| `bench_homogeneous.py` | Correctness and throughput sweep for one beam width |
| `bench_mixed_width.py` | Homogeneous and alternating mixed-width sweeps on one engine |
| `bench_http_reference.py` | Equivalent sweep against an existing HTTP reference server |
| `compare_ordered_outputs.py` | Exact order/set, Jaccard, intersection, and top-K overlap |
| `profile_trie_head.py` | CPU/GPU profiler capture for an in-process engine |
| `smoke_mixed_width.py` | Graph-enabled smoke for multiple widths on one server |
| `bench_normalizer.py` | Full-vocabulary normalizer latency and memory experiment |
| `bench_shared_context_attention.py` | Native Triton versus an optional reference kernel |
| `bench_flashinfer_cascade_attention.py` | FlashInfer cascade/shared-prefix alternatives |

The serving scripts expect gzip JSONL input rows containing `input_ids`. The
valid-SID file is a sorted NumPy array of radix-encoded SID keys. Override
`--token-start`, `--codebook-size`, and `--sid-length` for the model being
tested.

## Mixed-width serving example

Shared-context beam attention is enabled before importing SGLang:

```bash
export CUDA_VISIBLE_DEVICES=0
export SGLANG_BEAM_SHARED_CONTEXT_ATTENTION=true

python benchmark/trie_head_beam/bench_mixed_width.py \
  --model /path/to/model-with-trie-output-head \
  --valid-sid-keys /path/to/valid_sid_keys.npy \
  --correctness-inputs /path/to/correctness.jsonl.gz \
  --performance-inputs /path/to/performance.jsonl.gz \
  --output-dir /path/to/results \
  --label trie-head \
  --widths 1000 2000 \
  --concurrency 1 2 4 6 8 10 \
  --include-mixed \
  --mixed-concurrency 4 10 \
  --max-running-requests 20020 \
  --physical-gpu 0
```

Use at least 200 ordered requests for correctness and 1,000 measured requests
per performance point after warmup. The script writes compact JSON summaries
and gzip JSONL ordered outputs.

## Homogeneous comparison

`bench_homogeneous.py` can preserve CUDA graphs or disable them for a matched
eager comparison:

```bash
python benchmark/trie_head_beam/bench_homogeneous.py \
  --model /path/to/model-with-trie-output-head \
  --valid-sid-keys /path/to/valid_sid_keys.npy \
  --correctness-inputs /path/to/correctness.jsonl.gz \
  --performance-inputs /path/to/performance.jsonl.gz \
  --output-dir /path/to/results \
  --beams 2000 \
  --concurrency 1 4 10 \
  --disable-cuda-graph \
  --physical-gpu 0
```

Compare two ordered output files with:

```bash
python benchmark/trie_head_beam/compare_ordered_outputs.py \
  --reference /path/to/reference.jsonl.gz \
  --candidate /path/to/candidate.jsonl.gz \
  --output /path/to/comparison.json
```

## Kernel experiments

The synthetic kernel scripts use the Qwen-style attention geometry from the
original experiment. `bench_shared_context_attention.py` additionally expects
`--cute-root` to point at a compatible reference implementation exposing
`interface.BeamDecodeAttn`.

```bash
export CUDA_VISIBLE_DEVICES=0

python benchmark/trie_head_beam/bench_shared_context_attention.py \
  --cute-root /path/to/reference/python \
  --batch 1 4 10 \
  --beam-width 2000 \
  --physical-gpu 0

python benchmark/trie_head_beam/bench_flashinfer_cascade_attention.py \
  --groups 1 4 10 \
  --beam-width 2000 \
  --physical-gpu 0
```
