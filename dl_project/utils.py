# utils.py - fungsi bantu yang dipakai banyak file
import json
import random
import re

import numpy as np

REFUSAL_TEXT = ("Maaf, pertanyaan ini di luar cakupan UU 14/2008, PP 61/2010, "
                "UU 25/2009, dan UU 27/2022.")

# pola untuk membaca sitasi seperti [UU 14/2008 Pasal 17]
CITE_RE = re.compile(r"\b(UU|PP)\s*(?:No\.?\s*)?(\d+)\s*/\s*(\d{4})\s*,?\s*Pasal\s+(\d+[A-Za-z]?)", re.I)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pasal_id(law_key, pasal):
    return f"{law_key}_p{str(pasal).upper()}"


def citation_text(doc):
    return f"[{doc['law']} Pasal {doc['pasal']}]"


def parse_citations(text):
    # mengubah teks sitasi menjadi daftar id pasal (tanpa duplikat)
    ids = []
    for m in CITE_RE.finditer(text):
        pid = f"{m.group(1).upper()}{m.group(2)}_{m.group(3)}_p{m.group(4).upper()}"
        if pid not in ids:
            ids.append(pid)
    return ids


def is_refusal(text):
    return "di luar cakupan" in text.lower()
