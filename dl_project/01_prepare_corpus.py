# 01_prepare_corpus.py - mengubah PDF undang-undang menjadi korpus (satu chunk per pasal)
import json
import re

from pypdf import PdfReader

from config import (CORPUS_PATH, CORPUS_RAW_PATH, LAWS, MANUAL_FIX_PATH, OUT,
                    RAW_PDF_DIR)
from utils import pasal_id, write_jsonl

# baris yang dianggap noise (header, footer, nomor halaman)
NOISE_PATTERNS = [
    r"^\s*-\s*\d+\s*-\s*$",
    r"^\s*(www\.|https?://|jdih).*$",
    r"^\s*PRESIDEN\s*$",
    r"^\s*REPUBLIK INDONESIA\s*$",
    r"^\s*PRESIDEN REPUBLIK INDONESIA\s*$",
    r"^\s*SALINAN\s*$",
    r"^\s*\d+\s*$",
]
# baris judul bab yang bocor ke chunk pasal sebelumnya
HEADING_RE = re.compile(
    r"^(BAB\s+[IVXLC]+\b|Bagian\s+(Kesatu|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh|"
    r"Kedelapan|Kesembilan|Kesepuluh)\b|Paragraf\s+\d+\b)")
PASAL_RE = re.compile(r"(?m)^\s*Pasal\s+(\d+[A-Z]?)\s*$")
ITEM_RE = re.compile(r"^(\(\d+\)|[a-z][\.\)]|\d+\.)\s")


def read_pdf(path):
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def remove_noise(text):
    lines = []
    for line in text.split("\n"):
        if any(re.match(p, line) for p in NOISE_PATTERNS):
            continue
        lines.append(line)
    return "\n".join(lines)


def fix_ocr(text):
    # perbaikan kesalahan OCR yang umum; tambahkan pola baru di sini
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\bPasa[1I]\b", "Pasal", text)
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'")
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    return text


def cut_heading(text):
    # membuang judul bab/bagian di akhir chunk
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if HEADING_RE.match(line.strip()):
            return "\n".join(lines[:i])
    return text


def join_lines(text):
    # menggabungkan baris biasa, ayat dan huruf tetap di baris baru
    paragraphs = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not paragraphs or ITEM_RE.match(line):
            paragraphs.append(line)
        else:
            paragraphs[-1] += " " + line
    return re.sub(r"[ \t]+", " ", "\n".join(paragraphs)).strip()


def split_by_pasal(text):
    # hasil: {nomor_pasal: isi}
    matches = list(PASAL_RE.finditer(text))
    result = {}
    duplicates = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        no = m.group(1)
        if no in result:
            duplicates.append(no)
            continue
        result[no] = text[m.end():end]
    return result, duplicates


def split_body_penjelasan(text):
    # penjelasan dimulai dari baris "PENJELASAN", dan dicari setelah 30% awal dokumen
    for m in re.finditer(r"(?m)^\s*PENJELASAN\b", text):
        if m.start() > 0.3 * len(text):
            pen = text[m.start():]
            k = re.search(r"(?i)pasal\s+demi\s+pasal", pen)
            if k:
                pen = pen[k.end():]
            return text[:m.start()], pen
    return text, ""


def build_chunks(full_text, law_key, law_name, clean=True):
    if clean:
        full_text = fix_ocr(remove_noise(full_text))
    body, pen = split_body_penjelasan(full_text)
    body_pasal, dup = split_by_pasal(body)
    pen_pasal, _ = split_by_pasal(pen)
    chunks = []
    for no, raw in body_pasal.items():
        if clean:
            text = join_lines(cut_heading(raw))
            pj = join_lines(cut_heading(pen_pasal.get(no, "")))
            if re.fullmatch(r"(?i)cukup jelas\.?", pj.strip()):
                pj = ""
        else:
            text = re.sub(r"\s+", " ", raw).strip()
            pj = re.sub(r"\s+", " ", pen_pasal.get(no, "")).strip()
        chunks.append({"id": pasal_id(law_key, no), "law_key": law_key, "law": law_name,
                       "pasal": no, "text": text, "penjelasan": pj})
    return chunks, dup


def check_chunks(chunks, law_key, expected, duplicates):
    # laporan masalah yang perlu dicek manual terhadap PDF
    notes = []
    if len(chunks) != expected:
        notes.append(f"{law_key}: jumlah pasal {len(chunks)}, diharapkan {expected}")
    if duplicates:
        notes.append(f"{law_key}: nomor pasal ganda {duplicates}")
    for c in chunks:
        if len(c["text"]) < 80:
            notes.append(f"{c['id']}: teks sangat pendek ({len(c['text'])} karakter)")
        elif not re.search(r"[\.;:]\s*$", c["text"]):
            notes.append(f"{c['id']}: teks tidak diakhiri tanda baca, mungkin terpotong")
    return notes


def main():
    all_clean, all_raw, report = [], [], []
    for key, info in LAWS.items():
        path = RAW_PDF_DIR / info["pdf"]
        if not path.exists():
            print(f"[lewati] {path} tidak ada")
            continue
        text = read_pdf(path)
        clean, dup = build_chunks(text, key, info["name"], clean=True)
        raw, _ = build_chunks(text, key, info["name"], clean=False)
        report += check_chunks(clean, key, info["expected"], dup)
        n_pj = sum(1 for c in clean if c["penjelasan"])
        print(f"{info['name']}: {len(clean)} pasal, {n_pj} punya penjelasan")
        all_clean += clean
        all_raw += raw

    # perbaikan manual (misalnya pasal yang terpotong) dari data/manual_fixes.json
    if MANUAL_FIX_PATH.exists():
        fixes = json.loads(MANUAL_FIX_PATH.read_text(encoding="utf-8"))
        for c in all_clean:
            if c["id"] in fixes:
                c.update(fixes[c["id"]])
                print(f"perbaikan manual diterapkan: {c['id']}")

    write_jsonl(CORPUS_PATH, all_clean)
    write_jsonl(CORPUS_RAW_PATH, all_raw)
    (OUT / "corpus_report.txt").write_text("\n".join(report) or "tidak ada catatan", encoding="utf-8")
    print(f"\ntotal {len(all_clean)} chunk disimpan ke {CORPUS_PATH}")
    print("catatan yang perlu dicek manual:" if report else "tidak ada catatan")
    for r in report:
        print(" -", r)


if __name__ == "__main__":
    main()
