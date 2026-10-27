# 06_eval_generation.py - membandingkan sistem 1-4 pada split val/test
import argparse
import random

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (CORPUS_PATH, GEN_ADAPTER_DIR, GEN_MODEL, OUT, RETRIEVER_DIR,
                    RETRIEVER_MODEL, SEED, TEST_PATH, TOP_K, USE_PENJELASAN, VAL_PATH)
from prompts import build_messages
from retrieval import BM25Retriever, DenseRetriever, evaluate_retrieval
from utils import is_refusal, parse_citations, read_jsonl, set_seed


def generate(model, tok, messages, max_new=220):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def score(records, valid_ids):
    # metrik generasi; hanya pertanyaan in-scope untuk citation, out-of-scope untuk refusal
    ins = [r for r in records if r["gold"]]
    oos = [r for r in records if not r["gold"]]
    acc = sum(1 for r in ins if set(r["cited"]) & set(r["gold"])) / max(len(ins), 1)
    n_cites = sum(len(r["cited"]) for r in ins)
    good_cites = sum(len(set(r["cited"]) & set(r["gold"])) for r in ins)
    halluc = sum(1 for r in ins if any(c not in valid_ids for c in r["cited"])) / max(len(ins), 1)
    return {
        "citation_acc": acc,
        "citation_precision": good_cites / max(n_cites, 1),
        "hallucinated_cite_rate": halluc,
        "refusal_oos": sum(1 for r in oos if is_refusal(r["answer"])) / max(len(oos), 1),
        "false_refusal": sum(1 for r in ins if is_refusal(r["answer"])) / max(len(ins), 1),
        "n_in": len(ins), "n_oos": len(oos),
    }


def run_system(model, tok, queries, corpus, retriever):
    records = []
    for q in queries:
        docs = None
        if retriever is not None:
            docs = [corpus[i] for i, _ in retriever.search(q["question"], TOP_K)]
        ans = generate(model, tok, build_messages(q["question"], docs, USE_PENJELASAN))
        records.append({"qid": q["qid"], "question": q["question"], "type": q["type"],
                        "gold": q["gold"], "answer": ans, "cited": parse_citations(ans)})
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    set_seed(SEED)
    corpus = read_jsonl(CORPUS_PATH)
    valid_ids = {d["id"] for d in corpus}
    queries = read_jsonl(VAL_PATH if args.split == "val" else TEST_PATH)
    if args.limit:
        queries = queries[:args.limit]

    bm25 = BM25Retriever(corpus, USE_PENJELASAN)
    dense = DenseRetriever.from_pretrained(RETRIEVER_MODEL, corpus, USE_PENJELASAN)
    ft_dense = None
    if RETRIEVER_DIR.exists():
        ft_dense = DenseRetriever.from_pretrained(RETRIEVER_DIR, corpus, USE_PENJELASAN)

    # metrik retrieval untuk tiap retriever
    ret_rows = [{"retriever": "BM25", **evaluate_retrieval(bm25, queries)},
                {"retriever": "Dense pretrained", **evaluate_retrieval(dense, queries)}]
    if ft_dense:
        ret_rows.append({"retriever": "Dense fine-tuned", **evaluate_retrieval(ft_dense, queries)})
    ret_df = pd.DataFrame(ret_rows).round(3)
    print(ret_df.to_string(index=False))
    ret_df.to_csv(OUT / f"retrieval_{args.split}.csv", index=False)

    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, quantization_config=bnb, device_map={"": 0})
    model.eval()

    results, all_records = [], {}

    def add(name, records):
        all_records[name] = records
        results.append({"sistem": name, **score(records, valid_ids)})
        print("selesai:", name)

    # sistem 1-3 memakai model dasar tanpa adapter
    add("1 LM tanpa retrieval", run_system(model, tok, queries, corpus, None))
    add("2 BM25 + LM pretrained (baseline)", run_system(model, tok, queries, corpus, bm25))
    add("3 Dense + LM pretrained", run_system(model, tok, queries, corpus, dense))
    if ft_dense:
        add("3b Dense fine-tuned + LM pretrained", run_system(model, tok, queries, corpus, ft_dense))

    # sistem dengan adapter LoRA
    if GEN_ADAPTER_DIR.exists():
        model = PeftModel.from_pretrained(model, str(GEN_ADAPTER_DIR))
        model.eval()
        add("4a Dense + LM adapter", run_system(model, tok, queries, corpus, dense))
        if ft_dense:
            add("4 Dense fine-tuned + LM adapter (usulan)", run_system(model, tok, queries, corpus, ft_dense))

    df = pd.DataFrame(results).round(3)
    print("\n", df.to_string(index=False))
    df.to_csv(OUT / f"generation_{args.split}.csv", index=False)

    # sampel jawaban sistem usulan untuk dinilai manual (key-fact correctness)
    last = list(all_records)[-1]
    sample = [r for r in all_records[last] if r["gold"]]
    random.seed(SEED)
    sample = random.sample(sample, min(20, len(sample)))
    pd.DataFrame([{**r, "gold": ", ".join(r["gold"]), "cited": ", ".join(r["cited"]),
                   "key_fact_benar(0/1)": ""} for r in sample]).to_csv(
        OUT / f"manual_scoring_{args.split}.csv", index=False)
    pd.DataFrame([{"sistem": k, **r, "gold": ", ".join(r["gold"]), "cited": ", ".join(r["cited"])}
                  for k, v in all_records.items() for r in v]).to_csv(
        OUT / f"all_answers_{args.split}.csv", index=False)


if __name__ == "__main__":
    main()
