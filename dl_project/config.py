# config.py - pengaturan umum proyek
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_PDF_DIR = DATA / "raw"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# lokasi file data
CORPUS_PATH = DATA / "corpus.jsonl"
CORPUS_RAW_PATH = DATA / "corpus_raw.jsonl"
QUERIES_PATH = DATA / "queries.jsonl"
TRAIN_PATH = DATA / "train.jsonl"
VAL_PATH = DATA / "val.jsonl"
TEST_PATH = DATA / "test.jsonl"
MANUAL_FIX_PATH = DATA / "manual_fixes.json"

# daftar undang-undang: kunci, nama tampilan, nama file pdf, jumlah pasal yang diharapkan
LAWS = {
    "UU14_2008": {"name": "UU 14/2008", "pdf": "UU_14_2008.pdf", "expected": 64},
    "PP61_2010": {"name": "PP 61/2010", "pdf": "PP_61_2010.pdf", "expected": 22},
    "UU25_2009": {"name": "UU 25/2009", "pdf": "UU_25_2009.pdf", "expected": 62},
    "UU27_2022": {"name": "UU 27/2022", "pdf": "UU_27_2022.pdf", "expected": 76},
}

# model
RETRIEVER_MODEL = "intfloat/multilingual-e5-small"
RETRIEVER_DIR = OUT / "retriever_finetuned"
GEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GEN_ADAPTER_DIR = OUT / "lora_adapter"

# pengaturan umum
SEED = 42
TOP_K = 3                 # jumlah pasal yang diberikan ke generator
USE_PENJELASAN = True     # penjelasan ikut dipakai atau tidak
TEMPERATURE = 0.05        # suhu untuk contrastive loss
