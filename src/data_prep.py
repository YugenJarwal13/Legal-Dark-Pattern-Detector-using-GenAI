import fitz  # PyMuPDF
import re


def load_pdf(path):
    doc = fitz.open(path)
    texts = []

    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        if text.strip():
            texts.append({
                "page": page_num + 1,
                "text": text
            })

    doc.close()
    return texts


def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk_text(documents, chunk_size=500, overlap=100):
    chunks = []

    for doc in documents:
        text = clean_text(doc["text"])

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            chunks.append({
                "text": chunk,
                "page": doc["page"]
            })

            start += chunk_size - overlap

    return chunks