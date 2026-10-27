# 03_eval_retrieval.py - evaluasi BM25 dan dense retriever pretrained (belum di-fine-tune)
import argparse

import pandas as pd

from config import (CORPUS_PATH, OUT, RETRIEVER_MODEL, TEST_PATH, USE_PENJELASAN, VAL_PATH)
from retrieval import BM25Retriever, DenseRetriever, evaluate_retrieval
from utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    corpus = read_jsonl(CORPUS_PATH)
    queries = read_jsonl(VAL_PATH if args.split == "val" else TEST_PATH)

    rows = []
    bm25 = BM25Retriever(corpus, USE_PENJELASAN)
    rows.append({"sistem": "BM25", **evaluate_retrieval(bm25, queries)})

    dense = DenseRetriever.from_pretrained(RETRIEVER_MODEL, corpus, USE_PENJELASAN)
    rows.append({"sistem": "Dense pretrained (e5-small)", **evaluate_retrieval(dense, queries)})

    df = pd.DataFrame(rows).round(3)
    print(df.to_string(index=False))
    df.to_csv(OUT / f"retrieval_{args.split}_pretrained.csv", index=False)


if __name__ == "__main__":
    main()
