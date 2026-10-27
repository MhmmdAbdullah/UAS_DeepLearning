# 04_finetune_retriever.py - fine-tuning embedding retriever dengan contrastive loss (loop sendiri)
import argparse
import csv
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from config import (CORPUS_PATH, OUT, RETRIEVER_DIR, RETRIEVER_MODEL, TEMPERATURE,
                    TRAIN_PATH, USE_PENJELASAN, VAL_PATH)
from retrieval import BM25Retriever, DenseRetriever, doc_text, embed_texts, evaluate_retrieval
from utils import read_jsonl, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = read_jsonl(CORPUS_PATH)
    id2doc = {d["id"]: d for d in corpus}
    train_q = [q for q in read_jsonl(TRAIN_PATH) if q["gold"]]
    val_q = read_jsonl(VAL_PATH)

    # hard negative dari BM25: pasal mirip yang bukan gold
    bm25 = BM25Retriever(corpus, USE_PENJELASAN)
    pairs = []
    for q in train_q:
        cand = [corpus[i]["id"] for i, _ in bm25.search(q["question"], 8)
                if corpus[i]["id"] not in q["gold"]][:5]
        for g in q["gold"]:
            pairs.append({"question": q["question"], "pos": g, "golds": set(q["gold"]), "neg_cands": cand})
    print(f"{len(pairs)} pasangan (pertanyaan, pasal) untuk training, device={device}")

    tok = AutoTokenizer.from_pretrained(RETRIEVER_MODEL)
    model = AutoModel.from_pretrained(RETRIEVER_MODEL).to(device)

    def val_scores():
        r = DenseRetriever(corpus, model, tok, device, USE_PENJELASAN, max_len=384)
        return evaluate_retrieval(r, val_q)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = max(len(pairs) // args.batch, 1)
    total = steps_per_epoch * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    # skor awal sebelum training (epoch 0)
    s0 = val_scores()
    best_mrr, best_epoch, bad = s0["MRR"], 0, 0
    history = [{"epoch": 0, "train_loss": None, "val_mrr": s0["MRR"], "val_hit1": s0["hit@1"]}]
    print(f"epoch 0 | val MRR {s0['MRR']:.3f} | hit@1 {s0['hit@1']:.3f}")
    model.save_pretrained(RETRIEVER_DIR)
    tok.save_pretrained(RETRIEVER_DIR)

    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(pairs)
        losses = []
        for i in range(0, len(pairs) - 1, args.batch):
            batch = pairs[i:i + args.batch]
            if len(batch) < 2:
                continue
            pos_ids = [p["pos"] for p in batch]
            neg_ids = [random.choice(p["neg_cands"]) if p["neg_cands"] else p["pos"] for p in batch]
            all_ids = pos_ids + neg_ids

            q_emb = embed_texts(model, tok, ["query: " + p["question"] for p in batch], args.max_len, device)
            p_emb = embed_texts(model, tok, ["passage: " + doc_text(id2doc[x], USE_PENJELASAN) for x in all_ids],
                                args.max_len, device)
            logits = q_emb @ p_emb.T / TEMPERATURE

            # pasal gold lain (selain kolom positifnya sendiri) ditutup supaya tidak dihitung negatif
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for r, p in enumerate(batch):
                for c, pid in enumerate(all_ids):
                    if c != r and pid in p["golds"]:
                        mask[r, c] = True
            logits = logits.masked_fill(mask, -1e4)

            loss = F.cross_entropy(logits, torch.arange(len(batch), device=device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            losses.append(loss.item())

        sc = val_scores()
        tr = sum(losses) / max(len(losses), 1)
        history.append({"epoch": epoch, "train_loss": tr, "val_mrr": sc["MRR"], "val_hit1": sc["hit@1"]})
        print(f"epoch {epoch} | train loss {tr:.4f} | val MRR {sc['MRR']:.3f} | hit@1 {sc['hit@1']:.3f}")

        if sc["MRR"] > best_mrr + 1e-4:
            best_mrr, best_epoch, bad = sc["MRR"], epoch, 0
            model.save_pretrained(RETRIEVER_DIR)
            tok.save_pretrained(RETRIEVER_DIR)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping: val MRR tidak naik lagi")
                break

    print(f"\nmodel terbaik dari epoch {best_epoch} (val MRR {best_mrr:.3f}) disimpan di {RETRIEVER_DIR}")
    if best_epoch == 0:
        print("fine-tuning tidak mengalahkan model pretrained; yang tersimpan adalah model pretrained")

    with open(OUT / "retriever_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_mrr", "val_hit1"])
        w.writeheader()
        w.writerows(history)

    ep = [h["epoch"] for h in history if h["train_loss"] is not None]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in history if h["train_loss"] is not None], marker="o")
    ax[0].set_title("Train loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot([h["epoch"] for h in history], [h["val_mrr"] for h in history], marker="o", color="tab:orange")
    ax[1].set_title("Validation MRR")
    ax[1].set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(OUT / "retriever_curve.png", dpi=150)


if __name__ == "__main__":
    main()
