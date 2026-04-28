"""
Correctness test: SGLang's constrained beam search vs PR #36's HF reference.

Uses PR #36's actual PrefixIndex + PrefixIndexLogitsProcessor as ground truth.
Runs both engines on the same prompt + checkpoint, then checks:

  1. (Hard) SGLang's output beams are 100% valid against the prefix index.
  2. (Soft) SGLang's top-K beams overlap with HF's top-K (same beam search,
     same constraint => high overlap modulo FP-noise tie-breaking).

If (1) fails, the constraint hook is broken. If (1) passes but (2) is low,
the hook fires but math is off.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, "/home/jobuser")

MODEL = "/shared/public/sharing/generative-discovery-modeling/candidate_sourcing/checkpoints/sft_stage_b"
INDEX = "/shared/public/data/herosourcing/avats/semantic-id-training/index/RQ-Kmeans_Index/prefix_index-v2.npz"
PROMPT = "Required: 5+ years backend engineering, Python, distributed systems."
BEAM = 50      # Small for fast correctness check
MAX_NEW = 3
TOP_K = 20


def fmt_prompt(tok):
    return tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def hf_constrained():
    """HF beam search with PR #36's PrefixIndexLogitsProcessor."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
    from prefix_index import PrefixIndex

    # Faithful inline of PR #36's PrefixIndexLogitsProcessor (per-beam mask).
    class PrefixIndexLogitsProcessor:
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
    idx = PrefixIndex(index_path=INDEX, codebook_size=8192,
                      num_codebook=3, tokenizer=tok)

    text = fmt_prompt(tok)
    inputs = tok(text, return_tensors="pt").to("cuda")
    L = inputs.input_ids.shape[1]
    proc = PrefixIndexLogitsProcessor(idx, num_beams=BEAM)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=MAX_NEW, num_beams=BEAM,
            num_return_sequences=BEAM, do_sample=False,
            return_dict_in_generate=True, output_scores=True,
            logits_processor=LogitsProcessorList([proc]),
        )
    seqs = []
    for i in range(out.sequences.shape[0]):
        seqs.append(tuple(out.sequences[i][L:].cpu().tolist()))

    # Validate against the index
    ok = sum(1 for s in seqs if _seq_valid(s, idx))
    return seqs, ok, idx


def sglang_constrained(idx, vocab_size):
    """SGLang beam search with our constraint hook."""
    os.environ["SGL_BEAM_CONSTRAINT_INDEX_PATH"] = INDEX
    os.environ["SGL_BEAM_CONSTRAINT_TOK_PATH"] = MODEL
    os.environ["SGL_BEAM_CONSTRAINT_VOCAB_SIZE"] = str(vocab_size)
    os.environ["SGL_BEAM_CONSTRAINT_PYTHONPATH"] = "/home/jobuser"

    import sglang as sgl
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    text = fmt_prompt(tok)
    eng = sgl.Engine(
        model_path=MODEL, enable_beam_search=True, disable_radix_cache=True,
        dtype="bfloat16", trust_remote_code=True,
    )
    out = eng.generate(text, sampling_params={"max_new_tokens": MAX_NEW, "n": BEAM})
    beams = out.get("meta_info", {}).get("beam_results", [])
    seqs = [tuple(b["output_ids"][:MAX_NEW]) for b in beams]
    eng.shutdown()
    ok = sum(1 for s in seqs if _seq_valid(s, idx))
    return seqs, ok


def _seq_valid(seq, idx):
    if len(seq) >= 1 and seq[0] not in idx.query():
        return False
    if len(seq) >= 2 and seq[1] not in idx.query(seq[0]):
        return False
    if len(seq) >= 3 and seq[2] not in idx.query(seq[0], seq[1]):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["hf", "sglang", "both"], default="both")
    args = ap.parse_args()

    if args.engine in ("hf", "both"):
        print("=== HF (constrained, ground truth) ===")
        t0 = time.time()
        hf_seqs, hf_ok, idx = hf_constrained()
        print(f"  ran in {time.time()-t0:.1f}s")
        print(f"  beams returned: {len(hf_seqs)}")
        print(f"  valid against prefix index: {hf_ok}/{len(hf_seqs)}")
        print(f"  first 3 HF beams: {hf_seqs[:3]}")
        # Save for sglang comparison
        with open("/tmp/hf_seqs.txt", "w") as f:
            for s in hf_seqs:
                f.write(",".join(map(str, s)) + "\n")

    if args.engine in ("sglang", "both"):
        print("\n=== SGLang (constrained, our patch) ===")
        # Need vocab_size; quickly load tokenizer
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        vocab_size = len(tok)
        # Need PR #36 idx for validation only
        sys.path.insert(0, "/home/jobuser")
        from prefix_index import PrefixIndex
        val_idx = PrefixIndex(index_path=INDEX, codebook_size=8192,
                              num_codebook=3, tokenizer=tok)
        t0 = time.time()
        sg_seqs, sg_ok = sglang_constrained(val_idx, vocab_size)
        print(f"  ran in {time.time()-t0:.1f}s")
        print(f"  beams returned: {len(sg_seqs)}")
        print(f"  valid against prefix index: {sg_ok}/{len(sg_seqs)}")
        print(f"  first 3 SG beams: {sg_seqs[:3]}")

        # Compare with HF if both ran
        try:
            with open("/tmp/hf_seqs.txt") as f:
                hf_seqs = [tuple(int(x) for x in line.split(","))
                           for line in f if line.strip()]
            inter = set(hf_seqs) & set(sg_seqs)
            print(f"\n=== Overlap ===")
            print(f"  HF set: {len(set(hf_seqs))} unique")
            print(f"  SG set: {len(set(sg_seqs))} unique")
            print(f"  intersection: {len(inter)}/{BEAM} = {len(inter)/BEAM:.1%}")
            print(f"  top-{TOP_K} set overlap: "
                  f"{len(set(hf_seqs[:TOP_K]) & set(sg_seqs[:TOP_K]))}/{TOP_K}")
            print(f"  top-1 match: {hf_seqs[0] == sg_seqs[0]}")
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
