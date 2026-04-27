import os
from sentence_transformers import SentenceTransformer
import chromadb


class GDPRRAG:
    def __init__(self, file_path="data/gdpr.txt"):
        self.file_path = file_path
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name="gdpr_v4")

        # Only load once
        if self.collection.count() == 0:
            print("Building GDPR vector DB...")
            self._build_db()
        else:
            print("GDPR DB already loaded")

    # -----------------------------
    # STEP 1 — LOAD + CLEAN
    # -----------------------------
    def _load_text(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    # -----------------------------
    # STEP 2 — CHUNKING
    # -----------------------------
    def _chunk_text(self, text, chunk_size=200):
        """
        Sentence-aware chunking with size control and semantic filtering.
        - Keeps chunks short (LLM-friendly)
        - Avoids cutting sentences mid-way
        - Filters for relevance (Articles + key GDPR concepts)
        """
        import re

        # Split into sentences
        sentences = re.split(r'(?<=[.!?]) +', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Build chunk gradually
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += " " + sentence
            else:
                chunk_clean = current_chunk.strip()

                # FILTER CONDITIONS (IMPORTANT)
                if 50 < len(chunk_clean) < 400 and (
                    "Article" in chunk_clean or
                    "data" in chunk_clean.lower() or
                    "consent" in chunk_clean.lower() or
                    "processing" in chunk_clean.lower()
                ):
                    chunks.append(chunk_clean)

                # Start new chunk
                current_chunk = sentence

        # Handle last chunk
        if current_chunk:
            chunk_clean = current_chunk.strip()
            if 50 < len(chunk_clean) < 400:
                chunks.append(chunk_clean)

        return chunks

    # -----------------------------
    # STEP 3 — BUILD VECTOR DB
    # -----------------------------
    def _build_db(self):
        text = self._load_text()
        chunks = self._chunk_text(text)

        embeddings = self.model.encode(chunks).tolist()

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[str(i) for i in range(len(chunks))]
        )

        print(f"Stored {len(chunks)} GDPR chunks")

    # -----------------------------
    # STEP 4 — RETRIEVE
    # -----------------------------
    def retrieve(self, query, k=3):
        query_embedding = self.model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=6   # get more candidates
        )

        docs = results["documents"][0]

        # 🔥 SMART RANKING
        def score(chunk):
            score = 0

            chunk_lower = chunk.lower()

            # boost legal strength
            if "article" in chunk:
                score += 3

            # boost relevance keywords
            if "consent" in chunk_lower:
                score += 2
            if "processing" in chunk_lower:
                score += 2
            if "data" in chunk_lower:
                score += 1

            # penalize generic text
            if "scale" in chunk_lower or "technological developments" in chunk_lower:
                score -= 1

            return score

        ranked = sorted(docs, key=score, reverse=True)

        # clean + return top k
        return [r.strip().replace("\n", " ") for r in ranked[:k]]