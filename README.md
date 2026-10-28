# Adapting Language Models for Indonesian Public Information and Service Law Question Answering

Proyek UAS Deep Learning (Session 16), **Kelompok 2**, Track C: Foundation Models, Adaptation and Retrieval.

Anggota: Muhammad Abdullah, Zulfikar Ditya Antariksa, Taufiqurrahman, Zaky Hafiedz.

## 1. Model apa ini?

Sistem tanya jawab hukum berbasis **RAG (Retrieval-Augmented Generation)**. Warga menulis pertanyaan dengan bahasa sehari-hari, lalu sistem:

1. **Mencari pasal** yang relevan dari empat peraturan: UU 14/2008 (Keterbukaan Informasi Publik), PP 61/2010, UU 25/2009 (Pelayanan Publik), dan UU 27/2022 (Pelindungan Data Pribadi).
2. **Menjawab singkat** berdasarkan pasal tersebut dengan sitasi, misalnya `[UU 14/2008 Pasal 40]`. Jika pertanyaan di luar cakupan, sistem menolak dengan kalimat baku.

Keluaran adalah informasi hukum, bukan nasihat hukum, dan sistem tidak memutuskan apakah suatu pelanggaran terjadi.

| Komponen | Model | Yang dilatih kelompok |
|---|---|---|
| Retriever | `intfloat/multilingual-e5-small` | Fine-tuning kontrastif (loop PyTorch sendiri), hard negative dari BM25, early stopping |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct` (4-bit NF4) | QLoRA rank 8 (loop PyTorch sendiri), 2,18 juta parameter terlatih (0,14%) |
| Baseline | BM25 + Qwen2.5-1.5B-Instruct tanpa adapter | Tidak ada |

## 2. Hasil utama (split test, 42 pertanyaan in-scope dan 10 out-of-scope, seed 42)

| Sistem | Citation accuracy | Citation precision | Tolak out-of-scope | Salah tolak |
|---|---|---|---|---|
| 1. LM tanpa retrieval | 0,000 | 0,000 | 0,900 | 0,405 |
| 2. BM25 + LM pretrained (**baseline**) | 0,262 | 0,579 | 0,700 | 0,024 |
| 3. Dense + LM pretrained | 0,238 | 0,588 | 0,500 | 0,000 |
| 3b. Dense fine-tuned + LM pretrained | 0,167 | 0,412 | 0,600 | 0,000 |
| 4a. Dense pretrained + LM adapter | 0,690 | 0,744 | 0,900 | 0,095 |
| **4. Dense fine-tuned + LM adapter (usulan)** | **0,714** | 0,652 | 0,900 | 0,024 |

Retrieval (test): BM25 hit@1 0,476 dan MRR 0,596; dense fine-tuned hit@1 0,643 dan MRR 0,735.

Catatan: satu seed, sehingga simpangan baku antar seed belum tersedia. Lihat laporan untuk interval kepercayaan, ablasi, dan analisis error.

## 3. Saved model artefact

| Artefak | Tautan |
|---|---|
| `retriever_finetuned` | https://drive.google.com/drive/folders/1VZ_S1RdS2wMs4oxJCqEe9iABLKSYgRLK?usp=sharing |
| `lora_adapter` | https://drive.google.com/drive/folders/1RrJ-9F0oGgOLjWNSifPmU-9bPTZB0A3t?usp=sharing |

Letakkan isinya di `dl_project/outputs/retriever_finetuned/` dan `dl_project/outputs/lora_adapter/`.

## 4. Struktur repositori

```
UAS_DeepLearning/
  README.md
  dl_project/
    config.py                  path, nama model, seed (42), TOP_K
    utils.py  prompts.py  retrieval.py
    01_prepare_corpus.py       PDF -> korpus per pasal
    02_split_queries.py        split train/val/test per kelompok pasal (70/15/15)
    03_eval_retrieval.py       baseline BM25 dan dense pretrained
    04_finetune_retriever.py   fine-tuning retriever
    05_train_generator.py      LoRA/QLoRA generator (GPU)
    06_eval_generation.py      evaluasi sistem 1-4 (GPU)
    07_ablation_retrieval.py   ablasi korpus dan penjelasan
    08_demo.py                 demo tanya jawab (GPU)
    requirements.txt
    data/                      corpus.jsonl, queries.jsonl, train/val/test.jsonl
    outputs/                   hasil csv, grafik, dan artefak model
```

File PDF peraturan tidak disertakan di repositori. Letakkan di `dl_project/data/raw/` dengan nama `UU_14_2008.pdf`, `PP_61_2010.pdf`, `UU_25_2009.pdf`, `UU_27_2022.pdf` jika ingin menjalankan langkah 1 dari awal.

## 5. Lingkungan

- Python 3.10 atau lebih baru.
- Langkah 1-4 dan 7 berjalan di CPU. Langkah 5, 6, dan 8 membutuhkan GPU (diuji di Google Colab T4).
- Pustaka utama: `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `rank_bm25`, `pypdf`, `pandas`, `matplotlib`.

```bash
git clone https://github.com/MhmmdAbdullah/UAS_DeepLearning.git
cd UAS_DeepLearning/dl_project
pip install -r requirements.txt
```

## 6. Mereproduksi angka utama (satu perintah)

Angka utama adalah **citation accuracy 0,714** (sistem 4) pada split test. Di Google Colab dengan runtime GPU T4:

```bash
# 1. siapkan lingkungan dan unduh artefak ke dl_project/outputs
pip install -q rank_bm25 pypdf peft bitsandbytes accelerate transformers gdown
git clone https://github.com/MhmmdAbdullah/UAS_DeepLearning.git
cd UAS_DeepLearning/dl_project
gdown --folder "https://drive.google.com/drive/folders/1VZ_S1RdS2wMs4oxJCqEe9iABLKSYgRLK" -O outputs/retriever_finetuned
gdown --folder "https://drive.google.com/drive/folders/1RrJ-9F0oGgOLjWNSifPmU-9bPTZB0A3t" -O outputs/lora_adapter

# 2. perintah tunggal yang menghasilkan angka utama
python 06_eval_generation.py --split test
```

Keluaran: `outputs/generation_test.csv` (baris `4 Dense fine-tuned + LM adapter (usulan)`, kolom `citation_acc`) dan `outputs/retrieval_test.csv`.

Jika `gdown` gagal mengunduh folder, unduh kedua folder secara manual dari tautan di bagian 3.

Seed 42 ditetapkan di `config.py` dan dekoding bersifat greedy. Perbedaan numerik kecil tetap mungkin muncul akibat versi pustaka atau GPU yang berbeda.

## 7. Menjalankan seluruh pipeline dari awal

Jalankan dari folder `dl_project`.

| Langkah | Perintah | Perangkat | Keluaran |
|---|---|---|---|
| 1 | `python 01_prepare_corpus.py` | CPU | `data/corpus.jsonl`, `data/corpus_raw.jsonl`, `outputs/corpus_report.txt` |
| 2 | `python 02_split_queries.py` | CPU | `data/train.jsonl`, `val.jsonl`, `test.jsonl` |
| 3 | `python 03_eval_retrieval.py --split val` | CPU | `outputs/retrieval_val_pretrained.csv` |
| 4 | `python 04_finetune_retriever.py` | CPU | `outputs/retriever_finetuned/`, `retriever_curve.png` |
| 5 | `python 05_train_generator.py` | GPU | `outputs/lora_adapter/`, `generator_curve.png` |
| 6 | `python 06_eval_generation.py --split val` | GPU | `outputs/generation_val.csv` |
| 7 | `python 07_ablation_retrieval.py` | CPU | `outputs/ablation_retrieval_val.csv` |
| Akhir | `python 03_eval_retrieval.py --split test` dan `python 06_eval_generation.py --split test` | CPU dan GPU | angka test untuk laporan |

Prinsip: split test dijalankan satu kali di akhir, setelah semua keputusan dibuat berdasarkan validation.

`data/queries.jsonl` berisi 350 pertanyaan (explanation 145, situation 105, cross-law 30, out-of-scope 70) beserta gold pasal. Format setiap baris: `qid`, `question`, `type`, `gold` (daftar id pasal, kosong untuk out-of-scope), `answer`, `group_id`.

## 8. Demo

```bash
python 08_demo.py "Apakah saya wajib ikut mediasi kalau bersengketa informasi dengan badan publik?"
```

Skrip menampilkan pasal yang diambil, jawaban, dan sitasi yang terbaca. Membutuhkan GPU dan artefak pada bagian 3.

## 9. Keterbatasan

- Test kecil (42 pertanyaan in-scope), satu seed, dan test tidak memuat pertanyaan cross-law.
- Pertanyaan dibuat dengan bantuan AI lalu diperiksa; gold berupa satu pasal sehingga jawaban yang masuk akal bisa dinilai salah.
- Korpus UU 27/2022 belum sempurna (74 dari 76 pasal terbaca, dan sebagian chunk bercampur teks penjelasan).
- Sistem bukan penasihat hukum dan dapat keliru memilih pasal yang mirip.

## 10. Referensi

Hu et al. (2021) LoRA, arXiv:2106.09685. Dettmers et al. (2023) QLoRA, arXiv:2305.14314. Lewis et al. (2020) Retrieval-augmented generation. Wang et al. (2024) Multilingual E5, arXiv:2402.05672. Qwen Team (2024) Qwen2.5 Technical Report, arXiv:2412.15115.
