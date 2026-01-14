import os
import json
import requests
import faiss
import numpy as np
from tqdm import tqdm
from text_chunker import WhisperTranscriber
import re
import unicodedata

class BGEVectorStore:
    def __init__(
        self,
        model_name="bge-m3",
        ollama_url="http://localhost:11434/api/embed",
        dim=1024,
        index_path="src/vector_store"
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.dim = dim
        self.index_path = index_path

        self.index = faiss.IndexFlatIP(dim)  # cosine similarity
        self.metadata = []

    def _embed_once(self, texts):

        response = requests.post(
            self.ollama_url,
            json={"model": self.model_name, "input": texts},
            timeout=300
        )

        if response.status_code != 200:
            raise RuntimeError(response.text)

        data = response.json()

        if "embeddings" in data:
            emb = data["embeddings"]
        elif "data" in data:
            emb = [d["embedding"] for d in data["data"]]
        else:
            raise KeyError(data)

        arr = np.array(emb, dtype="float32")

        if np.isnan(arr).any():
            raise ValueError("NaN detected")

        return arr

    import re
    def normalize_for_embedding(self, text: str) -> str:
        if not text:
            return ""

        # Normalize unicode (convert smart quotes, bullets, etc.)
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")

        # Remove bullets explicitly
        text = re.sub(r"[•▪▫◦]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()


    def clean_for_embedding(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # Remove pure numbers
        if text.isdigit():
            return ""

        # Remove dot leaders / separators
        if re.fullmatch(r"[.\-•\s]+", text):
            return ""

        # Remove very short lines
        if len(text) < 30:
            return ""

        # Remove lines with very low alphabet ratio
        alpha_ratio = sum(c.isalpha() for c in text) / len(text)
        if alpha_ratio < 0.3:
            return ""

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text

    # EMBEDDING
    def embed_batch(self, texts):
        try:
            return self._embed_once(texts)
        except Exception as e:
            print("⚠️ Batch failed, retrying individually:", e)

            embeddings = []
            for t in texts:
                try:
                    emb = self._embed_once([t])
                    embeddings.append(emb[0])
                except Exception:
                    print("❌ Skipping text (still NaN):", t[:80])

            if not embeddings:
                raise RuntimeError("All texts failed embedding")

            embeddings = np.array(embeddings, dtype="float32")
            faiss.normalize_L2(embeddings)
            return embeddings
    # def embed_batch(self, texts):
    #     response = requests.post(
    #         self.ollama_url,
    #         json={
    #             "model": self.model_name,
    #             "input": texts
    #         },
    #         timeout=300
    #     )
    #
    #     print("\n\n")
    #     print(response.json())
    #     print(response)
    #     embeddings = np.array(response.json()["embeddings"], dtype="float32")
    #
    #     # Normalize for cosine similarity
    #     faiss.normalize_L2(embeddings)
    #     return embeddings

    # INGEST
    def add_documents(self, records, batch_size=32):
        """
        records: list of dicts with 'embedding_text'
        """
        for i in tqdm(range(0, len(records), batch_size), desc="Embedding"):
            batch = records[i:i + batch_size]
            # texts = [r["embedding_text"] for r in batch]

            texts = []
            valid_records = []

            for r in batch:
                t = self.normalize_for_embedding(r["embedding_text"])

                if len(t) < 30:
                    continue

                texts.append(t)
                valid_records.append(r)

            if not texts:
                continue
            print("\n\n")
            print(texts)

            embeddings = self.embed_batch(texts)

            self.index.add(embeddings)
            self.metadata.extend(batch)

    # SAVE
    def save(self):
        os.makedirs(self.index_path, exist_ok=True)

        faiss.write_index(
            self.index,
            os.path.join(self.index_path, "index.faiss")
        )

        with open(os.path.join(self.index_path, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    # LOAD
    def load(self):
        self.index = faiss.read_index(
            os.path.join(self.index_path, "index.faiss")
        )

        with open(os.path.join(self.index_path, "metadata.json"), "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    # SEARCH
    def search(self, query, top_k=5):
        query_embedding = self.embed_batch([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            record = self.metadata[idx].copy()
            record["score"] = float(score)
            results.append(record)

        return results

    def search_vector_db(self, query: str, top_k: int = 5,score_threshold: float = 0.25):
        # Load vector database
        db = BGEVectorStore(index_path="src/vector_db")
        db.load()

        # Perform semantic search
        results = db.search(query, top_k=top_k)

        # Filter weak matches
        filtered_results = [
            r for r in results
            if r["score"] >= score_threshold
        ]

        return filtered_results


# #  RUN
# transcriber = WhisperTranscriber()
# chunks = transcriber.ingest("C://Users//rutur/Downloads/The Ultimate C Handbook.pdf")
# print(chunks)
# vector_db = BGEVectorStore(model_name="bge-m3", dim=1024, index_path="src/vector_db")
# vector_db.add_documents(chunks, batch_size=32)
# vector_db.save()