# 02_split_queries.py - membagi pertanyaan ke train/val/test per kelompok pasal
import random
from collections import Counter, defaultdict

from config import (CORPUS_PATH, QUERIES_PATH, SEED, TEST_PATH, TRAIN_PATH, VAL_PATH)
from utils import read_jsonl, write_jsonl


def make_groups(queries):
    # pertanyaan yang berbagi pasal gold digabung ke satu kelompok (union-find sederhana)
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for q in queries:
        for g in q["gold"]:
            parent.setdefault(g, g)
        for g in q["gold"][1:]:
            union(q["gold"][0], g)
    groups = defaultdict(list)
    for q in queries:
        if q["gold"]:
            groups["pasal_" + find(q["gold"][0])].append(q)
        else:
            groups["oos_" + str(q.get("group_id", q["qid"]))].append(q)
    return list(groups.values())


def jaccard(a, b):
    a, b = set(a.lower().split()), set(b.lower().split())
    return len(a & b) / max(len(a | b), 1)


def ngrams(text, n=8):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    queries = read_jsonl(QUERIES_PATH)
    corpus = {d["id"]: d for d in read_jsonl(CORPUS_PATH)}
    for q in queries:
        for g in q["gold"]:
            assert g in corpus, f"gold {g} pada {q['qid']} tidak ada di korpus"

    random.seed(SEED)
    groups = make_groups(queries)
    split = {"train": [], "val": [], "test": []}
    ratio = {"train": 0.70, "val": 0.15, "test": 0.15}
    # kelompok in-scope dan out-of-scope dibagi terpisah agar out-of-scope ada di setiap split
    for part in ([g for g in groups if g[0]["gold"]], [g for g in groups if not g[0]["gold"]]):
        random.shuffle(part)
        part.sort(key=len, reverse=True)   # kelompok terbesar dibagi lebih dulu
        n_part = sum(len(g) for g in part)
        target = {name: r * n_part for name, r in ratio.items()}
        got = {name: 0 for name in ratio}
        for g in part:
            # kelompok masuk ke split yang paling jauh dari targetnya
            name = max(target, key=lambda s: target[s] - got[s])
            split[name] += g
            got[name] += len(g)

    # kelompok terbesar, untuk memeriksa apakah ada kelompok yang terlalu besar
    sizes = sorted((len(g) for g in groups), reverse=True)[:5]
    print("ukuran 5 kelompok terbesar:", sizes)

    for name, rows in split.items():
        print(f"{name}: {len(rows)} pertanyaan", dict(Counter(q["type"] for q in rows)))

    # cek kemiripan pertanyaan antar split
    print("\ncek near-duplicate antar split (Jaccard >= 0.7):")
    found = 0
    names = ["train", "val", "test"]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in split[names[i]]:
                for b in split[names[j]]:
                    if jaccard(a["question"], b["question"]) >= 0.7:
                        found += 1
                        print(f"  {a['qid']} ({names[i]}) mirip {b['qid']} ({names[j]})")
    print("  tidak ada" if found == 0 else f"  ditemukan {found} pasang, periksa manual")

    # cek jawaban train tidak memuat teks pasal yang menjadi gold test
    test_gold = {g for q in split["test"] for g in q["gold"]}
    leak = 0
    for q in split["train"]:
        ans = q.get("answer", "")
        if not ans:
            continue
        ans_grams = ngrams(ans)
        for g in test_gold:
            if ans_grams & ngrams(corpus[g]["text"]):
                leak += 1
                print(f"  kebocoran: jawaban {q['qid']} memuat teks {g}")
    print("cek kebocoran jawaban train:", "aman" if leak == 0 else f"{leak} kasus")

    write_jsonl(TRAIN_PATH, split["train"])
    write_jsonl(VAL_PATH, split["val"])
    write_jsonl(TEST_PATH, split["test"])


if __name__ == "__main__":
    main()
    