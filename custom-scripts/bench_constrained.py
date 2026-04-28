"""
HF vs SGLang CONSTRAINED beam-search benchmark using PR #36's setup verified
in test_correctness.py.

Identical config to the unconstrained bench:
  Model: SFT Qwen3-0.6B (sft_stage_b)
  Index: RQ-Kmeans_Index/prefix_index-v2.npz
  Beam=500, max_new=3, num_return=200
  16 prompts, sequential per-prompt loop, 2 runs

HF uses PR #36's faithful PrefixIndexLogitsProcessor.
SGLang uses our patched beam search + PrefixMaskProcessor (env-var-loaded).
Both produce 100% valid SIDs (verified separately).
"""
import argparse, os, sys, time
import numpy as np

MODEL = os.environ.get(
    "SGL_TEST_MODEL_PATH",
    "/shared/public/sharing/generative-discovery-modeling/"
    "candidate_sourcing/checkpoints/sft_stage_b",
)
INDEX = os.environ.get(
    "SGL_TEST_PREFIX_INDEX",
    "/shared/public/data/herosourcing/avats/semantic-id-training/"
    "index/RQ-Kmeans_Index/prefix_index-v2.npz",
)
INDEX_PY_DIR = os.environ.get(
    "SGL_TEST_INDEX_PY_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
sys.path.insert(0, INDEX_PY_DIR)

JOB_PROMPTS = [
    "Required: 5+ years backend engineering, Python, distributed systems.",
    "Required: PhD or MS in ML with 3+ years industry experience.",
    "Required: Senior frontend engineer with React, TypeScript expertise.",
    "Required: SRE/DevOps with deep Kubernetes and observability expertise.",
    "Required: Senior data engineer with Spark, Airflow, dbt at scale.",
    "Required: Security engineer with offensive security background.",
    "Required: ML engineer specializing in recommendation systems.",
    "Required: Senior PM with B2B SaaS experience.",
    "Required: Compiler engineer with strong LLVM experience.",
    "Required: Staff engineer with distributed-database expertise.",
    "Required: Research scientist with computer-vision background.",
    "Required: Developer relations with technical-writing background.",
    "Required: Senior product designer with motion design.",
    "Required: Hardware engineer with ASIC verification experience.",
    "Required: Senior TPM for ML infrastructure programs.",
    "Required: SRE for AI inference platforms on GPU clusters.",
]


def fmt(tok, p):
    return tok.apply_chat_template(
        [{"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def bench_hf(num_beams, max_new, num_return, n_runs):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
    from prefix_index import PrefixIndex

    class PrefixIndexLogitsProcessor:
        """Faithful copy of PR #36's processor (per-beam mask)."""
        def __init__(self, index, num_beams):
            self._index = index
            self._num_beams = num_beams
            self.count = 0

        def __call__(self, input_ids, scores):
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            mask = torch.full_like(scores, -1000000.0)
            B = input_ids.view(-1, self._num_beams, input_ids.shape[-1])
            for batch_id, beam_sent in enumerate(B):
                for beam_id, sent in enumerate(beam_sent):
                    if self.count == 0:
                        hash_key = []
                    else:
                        hash_key = sent[-self.count:].tolist()
                    if len(hash_key) < self._index.num_codebook:
                        allowed = self._index.query(*hash_key)
                    else:
                        allowed = []
                    if not allowed:
                        continue
                    mask[batch_id * self._num_beams + beam_id, allowed] = 0
            self.count += 1
            return scores + mask

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda").eval()
    idx = PrefixIndex(index_path=INDEX, codebook_size=8192, num_codebook=3,
                      tokenizer=tok)
    formatted = [fmt(tok, p) for p in JOB_PROMPTS]

    # Warmup
    inp0 = tok(formatted[0], return_tensors="pt").to("cuda")
    p_warm = PrefixIndexLogitsProcessor(idx, num_beams=4)
    with torch.no_grad():
        model.generate(**inp0, max_new_tokens=max_new, num_beams=4,
                       num_return_sequences=4, do_sample=False,
                       logits_processor=LogitsProcessorList([p_warm]))
    torch.cuda.synchronize()

    per_prompt = []; total_per_run = []
    for r in range(n_runs):
        torch.cuda.synchronize()
        run_t0 = time.time()
        for f in formatted:
            t0 = time.time()
            inputs = tok(f, return_tensors="pt").to("cuda")
            proc = PrefixIndexLogitsProcessor(idx, num_beams=num_beams)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new, num_beams=num_beams,
                    num_return_sequences=num_return,
                    do_sample=False, temperature=1.0,
                    output_scores=True, return_dict_in_generate=True,
                    logits_processor=LogitsProcessorList([proc]),
                )
            L = inputs.input_ids.shape[1]
            for i in range(len(out.sequences)):
                _ = tok.decode(out.sequences[i][L:], skip_special_tokens=True)
            torch.cuda.synchronize()
            per_prompt.append(time.time() - t0)
        total_per_run.append(time.time() - run_t0)
    return per_prompt, total_per_run


def bench_sglang(num_beams, max_new, num_return, n_runs):
    import sglang as sgl
    from transformers import AutoTokenizer

    # Activate the application-side processor through SGLang's standard
    # CustomLogitProcessor framework — same dispatch as DisallowedTokens-
    # LogitsProcessor / ThinkingBudgetLogitProcessor.
    app_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, app_dir)
    from prefix_index_processor import PrefixIndexCustomLogitProcessor

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    formatted = [fmt(tok, p) for p in JOB_PROMPTS]

    engine = sgl.Engine(
        model_path=MODEL, enable_beam_search=True, disable_radix_cache=True,
        enable_custom_logit_processor=True,
        dtype="bfloat16", trust_remote_code=True,
    )
    sp = {
        "max_new_tokens": max_new, "n": num_beams,
        "custom_params": {
            "prefix_index_module": "prefix_index",
            "prefix_index_class": "PrefixIndex",
            "prefix_index_kwargs": {
                "index_path": INDEX,
                "codebook_size": 8192,
                "num_codebook": 3,
            },
            "tokenizer_path": MODEL,
            "vocab_size": len(tok),
            "module_search_paths": [INDEX_PY_DIR, app_dir],
        },
    }
    clp = PrefixIndexCustomLogitProcessor.to_str()
    engine.generate(formatted[0], sampling_params={"max_new_tokens": max_new, "n": 4})

    per_prompt = []; total_per_run = []
    for r in range(n_runs):
        run_t0 = time.time()
        for f in formatted:
            t0 = time.time()
            out = engine.generate(f, sampling_params=sp,
                                  custom_logit_processor=clp)
            beams = out.get("meta_info", {}).get("beam_results", [])
            for b in beams[:num_return]:
                _ = b.get("text", "")
            per_prompt.append(time.time() - t0)
        total_per_run.append(time.time() - run_t0)
    engine.shutdown()
    return per_prompt, total_per_run


def summary(name, pp, tt, n_prompts):
    a = np.array(pp); t = np.array(tt)
    print(f"\n========================================")
    print(f"  {name}")
    print(f"========================================")
    print(f"  prompts/run={n_prompts}, runs={len(t)}")
    print(f"  TOTAL per run:  mean={t.mean():.4f}s  min={t.min():.4f}  max={t.max():.4f}")
    print(f"  PER-PROMPT:     mean={a.mean():.4f}s  std={a.std():.4f}  "
          f"min={a.min():.4f}  max={a.max():.4f}")
    print(f"  PER-PROMPT pct: p50={np.percentile(a,50):.4f}  "
          f"p90={np.percentile(a,90):.4f}  p99={np.percentile(a,99):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "sglang"], required=True)
    ap.add_argument("--num-beams", type=int, default=500)
    ap.add_argument("--max-new", type=int, default=3)
    ap.add_argument("--num-return", type=int, default=200)
    ap.add_argument("--n-runs", type=int, default=2)
    args = ap.parse_args()

    print(f"engine={args.engine}  beam={args.num_beams}  "
          f"max_new={args.max_new}  num_return={args.num_return}  "
          f"n_runs={args.n_runs}  CONSTRAINED")

    if args.engine == "hf":
        pp, tt = bench_hf(args.num_beams, args.max_new, args.num_return,
                          args.n_runs)
        summary(f"HF constrained (Qwen3-0.6B SFT, beam={args.num_beams})",
                pp, tt, len(JOB_PROMPTS))
    else:
        pp, tt = bench_sglang(args.num_beams, args.max_new, args.num_return,
                              args.n_runs)
        summary(f"SGLang constrained (Qwen3-0.6B SFT, beam={args.num_beams})",
                pp, tt, len(JOB_PROMPTS))


if __name__ == "__main__":
    main()
