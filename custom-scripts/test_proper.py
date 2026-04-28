"""
Verify the framework-based PrefixIndexCustomLogitProcessor against PR #36 HF.

Uses SamplingParams.custom_params + engine.generate(custom_logit_processor=...)
— the standard SGLang dispatch — to apply per-beam prefix-trie constraints.

Defaults to the LinkedIn-internal stage-B checkpoint and PR #36 prefix index.
Override with --model / --index / --index-py-dir / --app-dir if running in a
different environment.
"""
import os, sys, time, argparse


def _default_path(env, fallback):
    return os.environ.get(env, fallback)


MODEL = _default_path(
    "SGL_TEST_MODEL_PATH",
    "/shared/public/sharing/generative-discovery-modeling/"
    "candidate_sourcing/checkpoints/sft_stage_b",
)
INDEX = _default_path(
    "SGL_TEST_PREFIX_INDEX",
    "/shared/public/data/herosourcing/avats/semantic-id-training/"
    "index/RQ-Kmeans_Index/prefix_index-v2.npz",
)
# Directory containing PR #36's `prefix_index.py`. Defaults to the directory
# the test itself lives in; override with SGL_TEST_INDEX_PY_DIR if PrefixIndex
# is shipped from a different package.
INDEX_PY_DIR = _default_path(
    "SGL_TEST_INDEX_PY_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
sys.path.insert(0, INDEX_PY_DIR)
PROMPT = "Required: 5+ years backend engineering, Python, distributed systems."
BEAM = 50
MAX_NEW = 3


def fmt(tok):
    return tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def hf_run():
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
                    hash_key = [] if self.count == 0 else sent[-self.count:].tolist()
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
    text = fmt(tok)
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
    return [tuple(out.sequences[i][L:].cpu().tolist())
            for i in range(out.sequences.shape[0])], idx


def sg_run(idx, app_dir):
    import sglang as sgl
    from transformers import AutoTokenizer
    # PrefixIndexCustomLogitProcessor lives in application-side code; the user
    # decides where it sits. Here we point at this file's own directory so the
    # processor module sits next to the test.
    sys.path.insert(0, app_dir)
    from prefix_index_processor import PrefixIndexCustomLogitProcessor

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    text = fmt(tok)
    eng = sgl.Engine(
        model_path=MODEL, enable_beam_search=True, disable_radix_cache=True,
        enable_custom_logit_processor=True,
        dtype="bfloat16", trust_remote_code=True,
    )
    # Tokenizer path is forwarded into PrefixIndex so its sid_token_offset can
    # be resolved on the worker; PrefixIndex accepts a `tokenizer` kwarg, but
    # because tokenizer instances aren't dill-friendly we pass the path and
    # build it on the worker side. (For simplicity here we omit it — the
    # PrefixIndex query path that we actually use does not depend on the
    # tokenizer; sid_token_offset is only used by separate index-build code.)
    sampling_params = {
        "max_new_tokens": MAX_NEW,
        "n": BEAM,
        "custom_params": {
            "prefix_index_module": "prefix_index",
            "prefix_index_class": "PrefixIndex",
            "prefix_index_kwargs": {
                "index_path": INDEX,
                "codebook_size": 8192,
                "num_codebook": 3,
            },
            # Forward to the worker so PrefixIndex can resolve sid_token_offset.
            "tokenizer_path": MODEL,
            "vocab_size": len(tok),
            "module_search_paths": [INDEX_PY_DIR, app_dir],
        },
    }
    out = eng.generate(
        text,
        sampling_params=sampling_params,
        custom_logit_processor=PrefixIndexCustomLogitProcessor.to_str(),
    )
    beams = out.get("meta_info", {}).get("beam_results", [])
    seqs = [tuple(b["output_ids"][:MAX_NEW]) for b in beams]
    eng.shutdown()
    return seqs


def is_valid(seq, idx):
    if len(seq) >= 1 and seq[0] not in idx.query(): return False
    if len(seq) >= 2 and seq[1] not in idx.query(seq[0]): return False
    if len(seq) >= 3 and seq[2] not in idx.query(seq[0], seq[1]): return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hf", "sglang", "compare"], required=True)
    args = ap.parse_args()
    if args.mode == "hf":
        seqs, _ = hf_run()
        with open("/tmp/hf_proper.txt", "w") as f:
            for s in seqs: f.write(",".join(map(str, s)) + "\n")
        print(f"HF: {len(seqs)} beams saved")
    elif args.mode == "sglang":
        from prefix_index import PrefixIndex
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        idx = PrefixIndex(index_path=INDEX, codebook_size=8192, num_codebook=3,
                          tokenizer=tok)
        app_dir = os.path.dirname(os.path.abspath(__file__))
        t0 = time.time()
        seqs = sg_run(idx, app_dir)
        print(f"SG: {len(seqs)} beams in {time.time()-t0:.1f}s")
        with open("/tmp/sg_proper.txt", "w") as f:
            for s in seqs: f.write(",".join(map(str, s)) + "\n")
        ok = sum(1 for s in seqs if is_valid(s, idx))
        print(f"valid: {ok}/{len(seqs)}")
        print(f"first 3: {seqs[:3]}")
    else:
        from prefix_index import PrefixIndex
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        idx = PrefixIndex(index_path=INDEX, codebook_size=8192, num_codebook=3,
                          tokenizer=tok)
        hf = [tuple(int(x) for x in line.split(","))
              for line in open("/tmp/hf_proper.txt") if line.strip()]
        sg = [tuple(int(x) for x in line.split(","))
              for line in open("/tmp/sg_proper.txt") if line.strip()]
        hf_ok = sum(1 for s in hf if is_valid(s, idx))
        sg_ok = sum(1 for s in sg if is_valid(s, idx))
        inter = set(hf) & set(sg)
        print(f"HF valid: {hf_ok}/{len(hf)}")
        print(f"SG valid: {sg_ok}/{len(sg)}")
        print(f"top-1 match: {hf[0] == sg[0]}")
        print(f"top-{BEAM} set overlap: {len(inter)}/{BEAM} = {len(inter)/BEAM:.0%}")


if __name__ == "__main__":
    main()
