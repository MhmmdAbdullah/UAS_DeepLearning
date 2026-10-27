# 07_ablation_retrieval.py - ablasi: teks mentah vs bersih, dan dengan/tanpa penjelasan
import pandas as pd

from config import CORPUS_PATH, CORPUS_RAW_PATH, OUT, RETRIEVER_MODEL, VAL_PATH
from retrieval import BM25Retriever, DenseRetriever, evaluate_retrieval
from utils import read_jsonl


def main():
    queries = read_jsonl(VAL_PATH)
    corpora = {"bersih": read_jsonl(CORPUS_PATH), "mentah": read_jsonl(CORPUS_RAW_PATH)}
    rows = []
    for cname, corpus in corpora.items():
        for use_pj in (True, False):
            setting = {"korpus": cname, "penjelasan": "ya" if use_pj else "tidak"}
            bm25 = BM25Retriever(corpus, use_pj)
            rows.append({**setting, "retriever": "BM25", **evaluate_retrieval(bm25, queries)})
            dense = DenseRetriever.from_pretrained(RETRIEVER_MODEL, corpus, use_pj)
            rows.append({**setting, "retriever": "Dense pretrained", **evaluate_retrieval(dense, queries)})
    df = pd.DataFrame(rows).round(3)
    print(df.to_string(index=False))
    df.to_csv(OUT / "ablation_retrieval_val.csv", index=False)


if __name__ == "__main__":
    main()
