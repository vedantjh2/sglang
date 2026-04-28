"""Full 16-prompt × beam=500 correctness + bench using the framework path.

Override paths via env vars:
  SGL_TEST_MODEL_PATH    — model checkpoint dir
  SGL_TEST_PREFIX_INDEX  — prefix_index-v2.npz file
  SGL_TEST_INDEX_PY_DIR  — directory containing prefix_index.py
"""
import os, sys, time, pickle, argparse
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
NUM_BEAMS = 500
MAX_NEW = 3
NUM_RETURN = 200

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


def run_sglang(n_runs=2):
    import sglang as sgl
    from transformers import AutoTokenizer
    # PrefixIndexCustomLogitProcessor lives next to this test file.
    app_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, app_dir)
    from prefix_index_processor import PrefixIndexCustomLogitProcessor

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    formatted = [fmt(tok, p) for p in JOB_PROMPTS]
    eng = sgl.Engine(
        model_path=MODEL, enable_beam_search=True, disable_radix_cache=True,
        enable_custom_logit_processor=True,
        dtype="bfloat16", trust_remote_code=True,
    )
    sp_warm = {"max_new_tokens": MAX_NEW, "n": 4}
    eng.generate(formatted[0], sampling_params=sp_warm)

    sp = {
        "max_new_tokens": MAX_NEW, "n": NUM_BEAMS,
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

    all_seqs = []
    per_prompt = []
    total_per_run = []
    for r in range(n_runs):
        run_t0 = time.time()
        for f in formatted:
            t0 = time.time()
            out = eng.generate(f, sampling_params=sp, custom_logit_processor=clp)
            beams = out.get("meta_info", {}).get("beam_results", [])
            seqs = [tuple(b["output_ids"][:MAX_NEW]) for b in beams[:NUM_RETURN]]
            if r == 0:
                all_seqs.append(seqs)
            for b in beams[:NUM_RETURN]:
                _ = b.get("text", "")
            per_prompt.append(time.time() - t0)
        total_per_run.append(time.time() - run_t0)
        print(f"  run {r+1}: total {total_per_run[-1]:.2f}s")
    eng.shutdown()
    return all_seqs, per_prompt, total_per_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sglang", "compare"], required=True)
    ap.add_argument("--n-runs", type=int, default=2)
    args = ap.parse_args()

    if args.mode == "sglang":
        seqs, pp, tt = run_sglang(args.n_runs)
        with open("/tmp/sg_proper_full.pkl", "wb") as f:
            pickle.dump({"seqs": seqs, "per_prompt": pp, "total_per_run": tt}, f)
        a = np.array(pp); t = np.array(tt)
        print(f"\nSGLang constrained (framework, {NUM_BEAMS}-beam, 16 prompts × {len(tt)} runs):")
        print(f"  total/run mean={t.mean():.4f}s")
        print(f"  per-prompt mean={a.mean():.4f}s std={a.std():.4f} p50={np.percentile(a,50):.4f}")
    else:
        from prefix_index import PrefixIndex
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        idx = PrefixIndex(index_path=INDEX, codebook_size=8192, num_codebook=3,
                          tokenizer=tok)
        with open("/tmp/hf_seqs.pkl", "rb") as f:
            hf = pickle.load(f)
        with open("/tmp/sg_proper_full.pkl", "rb") as f:
            data = pickle.load(f)
        sg = data["seqs"]

        def is_valid(s):
            if len(s) >= 1 and s[0] not in idx.query(): return False
            if len(s) >= 2 and s[1] not in idx.query(s[0]): return False
            if len(s) >= 3 and s[2] not in idx.query(s[0], s[1]): return False
            return True

        agg_hf = agg_sg = agg_inter = agg_total = 0
        agg_top1 = agg_top10 = agg_top50 = 0
        for hs, ss in zip(hf, sg):
            agg_hf += sum(1 for s in hs if is_valid(s))
            agg_sg += sum(1 for s in ss if is_valid(s))
            inter = set(hs) & set(ss)
            agg_inter += len(inter); agg_total += len(ss)
            agg_top1 += int(hs[0] == ss[0])
            agg_top10 += len(set(hs[:10]) & set(ss[:10]))
            agg_top50 += len(set(hs[:50]) & set(ss[:50]))
        n = len(hf)
        nhf = sum(len(s) for s in hf); nsg = sum(len(s) for s in sg)
        print(f"\n=== Full bench (framework path) ===")
        print(f"  HF valid:        {agg_hf}/{nhf} ({100*agg_hf/nhf:.1f}%)")
        print(f"  SG valid:        {agg_sg}/{nsg} ({100*agg_sg/nsg:.1f}%)")
        print(f"  set overlap:     {agg_inter}/{agg_total} ({100*agg_inter/agg_total:.1f}%)")
        print(f"  top-1 match:     {agg_top1}/{n} ({100*agg_top1/n:.0f}%)")
        print(f"  top-10 overlap:  {agg_top10/n:.1f}/10 ({100*agg_top10/(10*n):.0f}%)")
        print(f"  top-50 overlap:  {agg_top50/n:.1f}/50 ({100*agg_top50/(50*n):.0f}%)")


if __name__ == "__main__":
    main()
