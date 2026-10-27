# prompts.py - format prompt untuk generator
from utils import REFUSAL_TEXT, citation_text

SYSTEM_RAG = (
    "Kamu asisten informasi hukum Indonesia tentang keterbukaan informasi publik, "
    "pelayanan publik, dan pelindungan data pribadi. "
    "Jawab singkat dan hanya berdasarkan pasal pada konteks. "
    "Sebutkan pasal yang dipakai dengan format [UU 14/2008 Pasal 17]. "
    f"Jika pertanyaan di luar cakupan atau konteks tidak cukup, jawab persis: {REFUSAL_TEXT} "
    "Jawaban adalah informasi hukum, bukan nasihat hukum."
)

SYSTEM_NO_CONTEXT = (
    "Kamu asisten informasi hukum Indonesia tentang keterbukaan informasi publik, "
    "pelayanan publik, dan pelindungan data pribadi. "
    "Jawab singkat dan sebutkan pasal yang dipakai dengan format [UU 14/2008 Pasal 17]. "
    f"Jika pertanyaan di luar cakupan UU 14/2008, PP 61/2010, UU 25/2009, dan UU 27/2022, jawab persis: {REFUSAL_TEXT}"
)


def doc_context(doc, use_penjelasan=True, max_text=900, max_pen=300):
    # satu pasal dijadikan blok teks pendek untuk konteks
    s = f"{citation_text(doc)}\n{doc['text'][:max_text]}"
    if use_penjelasan and doc.get("penjelasan"):
        s += f"\nPenjelasan: {doc['penjelasan'][:max_pen]}"
    return s


def build_messages(question, docs=None, use_penjelasan=True):
    # docs=None berarti tanpa retrieval
    if docs is None:
        return [{"role": "system", "content": SYSTEM_NO_CONTEXT},
                {"role": "user", "content": f"Pertanyaan: {question}"}]
    ctx = "\n\n".join(doc_context(d, use_penjelasan) for d in docs)
    user = f"Konteks:\n{ctx}\n\nPertanyaan: {question}"
    return [{"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": user}]
