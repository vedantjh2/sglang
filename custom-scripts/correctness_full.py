"""
Full-config correctness check: same workload as the benchmark
(16 prompts, beam=500, max_new=3, num_return=200) on the SFT Qwen3-0.6B with
PR #36's actual prefix index.

Records per-prompt output token tuples to /tmp/{hf,sg}_seqs.npz, then computes:
  - validity (every output must respect the prefix index at each level)
  - per-prompt set overlap between HF and SGLang
  - top-1 / top-K rank-stability
"""
import argparse, os, sys, time, pickle
sys.path.insert(0, "/home/jobuser")

MODEL = "/shared/public/sharing/generative-discovery-modeling/candidate_sourcing/checkpoints/sft_stage_b"
INDEX = "/shared/public/data/herosourcing/avats/semantic-id-training/index/RQ-Kmeans_Index/prefix_index-v2.npz"
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


def run_hf():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
    from prefix_index import PrefixIndex

    class PrefixIndexLogitsProcessor:
        def __init__(self, index, num_beams):
            self._index = index; self._num_beams = num_beams; self.count = 0

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

    all_seqs = []
    for f in formatted:
        inputs = tok(f, return_tensors="pt").to("cuda")
        proc = PrefixIndexLogitsProcessor(idx, num_beams=NUM_BEAMS)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW, num_beams=NUM_BEAMS,
                num_return_sequences=NUM_RETURN, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
                logits_processor=LogitsProcessorList([proc]),
            )
        L = inputs.input_ids.shape[1]
        seqs = [tuple(out.sequences[i][L:].cpu().tolist())
                for i in range(out.sequences.shape[0])]
        all_seqs.append(seqs)
        print(f"  HF prompt {len(all_seqs)}/{len(formatted)}: {len(seqs)} beams")
    return all_seqs


def run_sglang():
    os.environ["SGL_BEAM_CONSTRAINT_INDEX_PATH"] = INDEX
    os.environ["SGL_BEAM_CONSTRAINT_TOK_PATH"] = MODEL
    os.environ["SGL_BEAM_CONSTRAINT_PYTHONPATH"] = "/home/jobuser"
    import sglang as sgl
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    os.environ["SGL_BEAM_CONSTRAINT_VOCAB_SIZE"] = str(len(tok))
    formatted = [fmt(tok, p) for p in JOB_PROMPTS]

    eng = sgl.Engine(
        model_path=MODEL, enable_beam_search=True, disable_radix_cache=True,
        dtype="bfloat16", trust_remote_code=True,
    )
    eng.generate(formatted[0], sampling_params={"max_new_tokens": MAX_NEW, "n": 4})
    sp = {"max_new_tokens": MAX_NEW, "n": NUM_BEAMS}

    all_seqs = []
    for f in formatted:
        out = eng.generate(f, sampling_params=sp)
        beams = out.get("meta_info", {}).get("beam_results", [])
        seqs = [tuple(b["output_ids"][:MAX_NEW]) for b in beams[:NUM_RETURN]]
        all_seqs.append(seqs)
        print(f"  SG prompt {len(all_seqs)}/{len(formatted)}: {len(seqs)} beams")
    eng.shutdown()
    return all_seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "sglang", "compare"], required=True)
    args = ap.parse_args()

    if args.mode == "hf":
        seqs = run_hf()
        with open("/tmp/hf_seqs.pkl", "wb") as f:
            pickle.dump(seqs, f)
        print(f"saved {sum(len(s) for s in seqs)} sequences to /tmp/hf_seqs.pkl")
    elif args.mode == "sglang":
        seqs = run_sglang()
        with open("/tmp/sg_seqs.pkl", "wb") as f:
            pickle.dump(seqs, f)
        print(f"saved {sum(len(s) for s in seqs)} sequences to /tmp/sg_seqs.pkl")
    else:
        # Load both and compare
        from prefix_index import PrefixIndex
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        idx = PrefixIndex(index_path=INDEX, codebook_size=8192, num_codebook=3,
                          tokenizer=tok)
        with open("/tmp/hf_seqs.pkl", "rb") as f:
            hf = pickle.load(f)
        with open("/tmp/sg_seqs.pkl", "rb") as f:
            sg = pickle.load(f)
        print(f"loaded {len(hf)} prompts, HF beams/prompt={len(hf[0])}, "
              f"SG beams/prompt={len(sg[0])}")

        def is_valid(seq):
            if len(seq) >= 1 and seq[0] not in idx.query(): return False
            if len(seq) >= 2 and seq[1] not in idx.query(seq[0]): return False
            if len(seq) >= 3 and seq[2] not in idx.query(seq[0], seq[1]): return False
            return True

        print("\n=== Per-prompt validity + overlap ===")
        print(f"{'idx':>3}  {'HF valid':>10}  {'SG valid':>10}  "
              f"{'set overlap':>12}  {'top-1 match':>12}  {'top-10 set':>12}  {'top-50 set':>12}")
        agg_hf_valid = agg_sg_valid = 0; agg_inter = agg_total = 0
        agg_top1 = 0; agg_top10 = 0; agg_top50 = 0
        for i, (hs, ss) in enumerate(zip(hf, sg)):
            hf_v = sum(1 for s in hs if is_valid(s))
            sg_v = sum(1 for s in ss if is_valid(s))
            inter = set(hs) & set(ss)
            top1 = (hs[0] == ss[0])
            top10 = len(set(hs[:10]) & set(ss[:10]))
            top50 = len(set(hs[:50]) & set(ss[:50]))
            print(f"{i:>3}  {hf_v:>5}/{len(hs):<4}  {sg_v:>5}/{len(ss):<4}  "
                  f"{len(inter):>5}/{len(ss):<5}({len(inter)/len(ss):.0%})  "
                  f"{str(top1):>12}  {top10:>5}/10      {top50:>5}/50")
            agg_hf_valid += hf_v; agg_sg_valid += sg_v
            agg_inter += len(inter); agg_total += len(ss)
            agg_top1 += int(top1); agg_top10 += top10; agg_top50 += top50

        n = len(hf)
        print(f"\n=== Aggregate ===")
        print(f"  HF valid:        {agg_hf_valid}/{sum(len(s) for s in hf)} "
              f"({100*agg_hf_valid/sum(len(s) for s in hf):.1f}%)")
        print(f"  SG valid:        {agg_sg_valid}/{sum(len(s) for s in sg)} "
              f"({100*agg_sg_valid/sum(len(s) for s in sg):.1f}%)")
        print(f"  set overlap:     {agg_inter}/{agg_total} "
              f"({100*agg_inter/agg_total:.1f}%)")
        print(f"  top-1 match:     {agg_top1}/{n} ({100*agg_top1/n:.0f}%)")
        print(f"  top-10 overlap:  {agg_top10/n:.1f}/10 ({100*agg_top10/(10*n):.0f}%)")
        print(f"  top-50 overlap:  {agg_top50/n:.1f}/50 ({100*agg_top50/(50*n):.0f}%)")


if __name__ == "__main__":
    main()
