# 05_train_generator.py - LoRA/QLoRA pada generator (Qwen2.5-1.5B-Instruct) dengan loop training sendiri
import argparse
import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          get_linear_schedule_with_warmup)

from config import (CORPUS_PATH, GEN_ADAPTER_DIR, GEN_MODEL, OUT, TOP_K, TRAIN_PATH,
                    USE_PENJELASAN, VAL_PATH)
from prompts import build_messages
from retrieval import BM25Retriever
from utils import REFUSAL_TEXT, citation_text, parse_citations, read_jsonl, set_seed

MAX_LEN = 1800


def make_target(q, id2doc):
    # jawaban target; memakai jawaban tulisan kelompok kalau ada, kalau tidak pakai template ekstraktif
    if not q["gold"]:
        return REFUSAL_TEXT
    cites = [citation_text(id2doc[g]) for g in q["gold"]]
    ans = q.get("answer", "").strip()
    if not ans:
        first = id2doc[q["gold"][0]]["text"][:300].rsplit(" ", 1)[0]
        ans = f"Ketentuan yang relevan terdapat pada {', '.join(cites)}. Isinya: {first}"
    if not parse_citations(ans):
        ans += f" Dasar hukum: {', '.join(cites)}."
    return ans


def make_examples(queries, corpus, id2doc, bm25, tok):
    examples, skipped = [], 0
    for q in queries:
        # konteks dari BM25 top-k; pasal gold dimasukkan paksa agar jawaban bisa didukung konteks
        ids = list(q["gold"])[:TOP_K]
        for i, _ in bm25.search(q["question"], 10):
            if len(ids) >= TOP_K:
                break
            if corpus[i]["id"] not in ids:
                ids.append(corpus[i]["id"])
        random.shuffle(ids)
        docs = [id2doc[x] for x in ids]
        prompt = tok.apply_chat_template(build_messages(q["question"], docs, USE_PENJELASAN),
                                         tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        a_ids = tok(make_target(q, id2doc) + tok.eos_token, add_special_tokens=False).input_ids
        if len(p_ids) + len(a_ids) > MAX_LEN:
            skipped += 1
            continue
        examples.append({"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids})
    if skipped:
        print(f"{skipped} contoh dilewati karena melebihi {MAX_LEN} token")
    return examples


def to_tensors(ex, device):
    ids = torch.tensor([ex["input_ids"]], device=device)
    lab = torch.tensor([ex["labels"]], device=device)
    return ids, lab


def eval_loss(model, examples, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for ex in examples:
            ids, lab = to_tensors(ex, device)
            with torch.autocast("cuda", dtype=torch.float16):
                total += model(input_ids=ids, labels=lab).loss.item()
    return total / max(len(examples), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(GEN_ADAPTER_DIR))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU tidak tersedia. Jalankan di Google Colab (Runtime > GPU T4).")
    set_seed(args.seed)
    device = "cuda"

    corpus = read_jsonl(CORPUS_PATH)
    id2doc = {d["id"]: d for d in corpus}
    bm25 = BM25Retriever(corpus, USE_PENJELASAN)

    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    train_ex = make_examples(read_jsonl(TRAIN_PATH), corpus, id2doc, bm25, tok)
    val_ex = make_examples(read_jsonl(VAL_PATH), corpus, id2doc, bm25, tok)
    print(f"contoh train: {len(train_ex)}, contoh val: {len(val_ex)}")

    # model dasar dimuat 4-bit (QLoRA)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, quantization_config=bnb, device_map={"": 0})
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.1,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(train_ex) + args.accum - 1) // args.accum
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, max(int(0.1 * total), 1), total)
    scaler = torch.cuda.amp.GradScaler()

    best_val, best_epoch, bad = eval_loss(model, val_ex, device), 0, 0
    print(f"epoch 0 | val loss {best_val:.4f}")
    history = [{"epoch": 0, "train_loss": None, "val_loss": best_val}]
    model.save_pretrained(args.out)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_ex)
        run_loss, n = 0.0, 0
        for step, ex in enumerate(train_ex):
            ids, lab = to_tensors(ex, device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, labels=lab).loss
            scaler.scale(loss / args.accum).backward()
            run_loss += loss.item()
            n += 1
            if (step + 1) % args.accum == 0 or step + 1 == len(train_ex):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()

        tr, va = run_loss / max(n, 1), eval_loss(model, val_ex, device)
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va})
        print(f"epoch {epoch} | train loss {tr:.4f} | val loss {va:.4f}")
        if va < best_val - 1e-4:
            best_val, best_epoch, bad = va, epoch, 0
            model.save_pretrained(args.out)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping: val loss tidak turun lagi")
                break

    print(f"\nadapter terbaik dari epoch {best_epoch} (val loss {best_val:.4f}) ada di {args.out}")

    with open(OUT / "generator_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(history)
    ep = [h["epoch"] for h in history if h["train_loss"] is not None]
    plt.figure(figsize=(5.5, 3.5))
    plt.plot(ep, [h["train_loss"] for h in history if h["train_loss"] is not None], marker="o", label="train")
    plt.plot([h["epoch"] for h in history], [h["val_loss"] for h in history], marker="o", label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "generator_curve.png", dpi=150)


if __name__ == "__main__":
    main()
