import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class ClaimVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None  # initialized on build/load
        self.claims = []  # store original claims for reference

    def build_index(self, df: pd.DataFrame):
        """
        Build FAISS index from dataframe of claims.
        """
        # Prefer cleaned_description if present to normalize spelling/noise
        texts = (df["cleaned_description"] if "cleaned_description" in df.columns else df["claim_description"]).tolist()
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Use inner product on normalized vectors to get cosine similarity behavior
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        self.claims = texts
        print(f"✅ Added {len(texts)} claims to FAISS index")

    def search(self, query: str, k: int = 3):
        """
        Search FAISS index for similar claims.
        """
        if self.index is None:
            return []
        q = query
        # Light normalization: lowercase
        if isinstance(q, str):
            q = q.lower()
        q_vec = self.model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D, I = self.index.search(q_vec, k)

        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.claims):
                results.append({
                    "claim_text": self.claims[idx],
                    "similarity": float(score)
                })
        return results

    def save(self, path: str):
        """
        Save FAISS index + claims.
        """
        if self.index is None:
            raise ValueError("Index is empty; build_index before saving.")
        faiss.write_index(self.index, path + ".faiss")
        with open(path + ".claims.txt", "w", encoding="utf-8") as f:
            for c in self.claims:
                f.write(c + "\n")

    def load(self, path: str):
        """
        Load FAISS index + claims.
        """
        self.index = faiss.read_index(path + ".faiss")
        with open(path + ".claims.txt", "r", encoding="utf-8") as f:
            self.claims = [line.strip() for line in f]
        print(f"✅ Loaded FAISS index with {len(self.claims)} claims")
