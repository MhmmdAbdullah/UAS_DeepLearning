# 08_demo.py - demo tanya jawab: pertanyaan, pasal yang diambil, lalu jawaban
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import CORPUS_PATH, GEN_ADAPTER_DIR, GEN_MODEL, RETRIEVER_DIR, TOP_K, USE_PENJELASAN
from prompts import build_messages
from retrieval import DenseRetriever
from utils import parse_citations, read_jsonl


def generate(model, tok, messages, max_new=220):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    corpus = read_jsonl(CORPUS_PATH)
    retriever = DenseRetriever.from_pretrained(RETRIEVER_DIR, corpus, USE_PENJELASAN)

    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, quantization_config=bnb, device_map={"": 0})
    model = PeftModel.from_pretrained(model, str(GEN_ADAPTER_DIR))
    model.eval()

    # pertanyaan dari argumen, atau diketik satu per satu
    questions = sys.argv[1:]
    while True:
        q = questions.pop(0) if questions else input("\nPertanyaan (kosongkan untuk keluar): ").strip()
        if not q:
            break
        docs = [corpus[i] for i, _ in retriever.search(q, TOP_K)]
        print("\nPasal yang diambil:")
        for d in docs:
            print(f"  - {d['law']} Pasal {d['pasal']}: {d['text'][:110]}...")
        ans = generate(model, tok, build_messages(q, docs, USE_PENJELASAN))
        print("\nJawaban:", ans)
        print("Sitasi terbaca:", parse_citations(ans) or "tidak ada")
        print("(Informasi hukum, bukan nasihat hukum.)")
        if not sys.stdin.isatty() and not questions:
            break


if __name__ == "__main__":
    main()