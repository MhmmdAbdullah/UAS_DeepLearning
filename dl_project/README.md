# Proyek Deep Learning Group 2 (Track C)

Urutan menjalankan (di Google Colab, GPU T4 hanya perlu untuk langkah 05 dan 06):

1. Taruh PDF di `data/raw/` dengan nama sesuai `config.py` (UU_14_2008.pdf, PP_61_2010.pdf, UU_25_2009.pdf, UU_27_2022.pdf)
2. `python 01_prepare_corpus.py`  lalu cek `outputs/corpus_report.txt` terhadap PDF
3. Buat `data/queries.jsonl` (lihat `data/queries_contoh_format.jsonl`)
4. `python 02_split_queries.py`
5. `python 03_eval_retrieval.py --split val`
6. `python 04_finetune_retriever.py`
7. `python 05_train_generator.py`  (GPU) (Colab)
8. `python 06_eval_generation.py --split val`   (cek dulu di val) (Colab)
9. `python 07_ablation_retrieval.py`
10. Terakhir saja: `python 03_eval_retrieval.py --split test` dan `python 06_eval_generation.py --split test`

Penjelasan lengkap ada di file Word panduan.
