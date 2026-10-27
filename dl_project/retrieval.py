# retrieval.py - BM25, dense retriever, dan metrik retrieval
import re

import numpy as np
from rank_bm25 import BM25Okapi

# stopword bahasa Indonesia yang sederhana
STOPWORDS = set("""yang dan di ke dari untuk dengan atau pada dalam adalah itu ini oleh sebagai
dapat tidak akan juga karena serta bagi para saya aku apa bagaimana kapan siapa apakah
bisa harus ada jika maka telah sudah lebih saat suatu setiap""".split())


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def doc_text(doc, use_penjelasan=True):
    t = f"{doc['law']} Pasal {doc['pasal']}. {doc['text']}"
    if use_penjelasan and doc.get("penjelasan"):
        t += " Penjelasan: " + doc["penjelasan"]
    return t


class BM25Retriever:
    def __init__(self, corpus, use_penjelasan=True):
        self.corpus = corpus
        self.bm25 = BM25Okapi([tokenize(doc_text(d, use_penjelasan)) for d in corpus])

    def search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]


def embed_texts(model, tokenizer, texts, max_len, device):
    # mean pooling lalu dinormalisasi; gradien tetap aktif kalau dipanggil saat training
    import torch.nn.functional as F
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    return F.normalize(emb, dim=-1)


class DenseRetriever:
    def __init__(self, corpus, model, tokenizer, device, use_penjelasan=True, max_len=384):
        import torch
        self.corpus = corpus
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.model.eval()
        texts = ["passage: " + doc_text(d, use_penjelasan) for d in corpus]
        parts = []
        with torch.no_grad():
            for i in range(0, len(texts), 16):
                parts.append(embed_texts(model, tokenizer, texts[i:i + 16], max_len, device).cpu())
        self.doc_emb = torch.cat(parts)

    @classmethod
    def from_pretrained(cls, path, corpus, use_penjelasan=True, max_len=384):
        import torch
        from transformers import AutoModel, AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(str(path))
        model = AutoModel.from_pretrained(str(path)).to(device)
        return cls(corpus, model, tok, device, use_penjelasan, max_len)

    def search(self, query, k=5):
        import torch
        with torch.no_grad():
            q = embed_texts(self.model, self.tokenizer, ["query: " + query], self.max_len, self.device).cpu()
        scores = (q @ self.doc_emb.T)[0]
        top = torch.topk(scores, min(k, len(self.corpus)))
        return [(int(i), float(s)) for s, i in zip(top.values, top.indices)]


def evaluate_retrieval(retriever, queries, ks=(1, 3, 5)):
    # hanya pertanyaan in-scope; hit@k dan MRR dihitung dari ranking seluruh korpus
    n_docs = len(retriever.corpus)
    ids = [d["id"] for d in retriever.corpus]
    hits = {k: 0 for k in ks}
    rr_sum, n = 0.0, 0
    for q in queries:
        if not q["gold"]:
            continue
        ranking = [ids[i] for i, _ in retriever.search(q["question"], n_docs)]
        gold = set(q["gold"])
        n += 1
        for k in ks:
            if gold & set(ranking[:k]):
                hits[k] += 1
        for pos, pid in enumerate(ranking, start=1):
            if pid in gold:
                rr_sum += 1.0 / pos
                break
    result = {f"hit@{k}": hits[k] / max(n, 1) for k in ks}
    result["MRR"] = rr_sum / max(n, 1)
    result["n"] = n
    return result
