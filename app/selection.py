"""Extractive selection via Maximal Marginal Relevance."""

from __future__ import annotations

import numpy as np

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore


def _cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between each row of matrix and a vector."""
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    vector_norm = np.linalg.norm(vector)
    matrix_norms[matrix_norms == 0] = 1.0
    if vector_norm == 0:
        vector_norm = 1.0
    return matrix.dot(vector) / (matrix_norms.flatten() * vector_norm)


class Selector:
    """Select representative text chunks using MMR."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = (
            SentenceTransformer(model_name) if SentenceTransformer is not None else None
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            return np.array(self.model.encode(texts, normalize_embeddings=True))

        embeddings = [self._fallback_embed(text) for text in texts]
        return np.stack(embeddings, axis=0) if embeddings else np.zeros((0, 26))

    @staticmethod
    def _fallback_embed(text: str) -> np.ndarray:
        vec = np.zeros(26, dtype=float)
        for character in text.lower():
            if "a" <= character <= "z":
                vec[ord(character) - ord("a")] += 1.0
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec

    def mmr(
        self, embeddings: np.ndarray, query_vec: np.ndarray, k: int, lam: float
    ) -> list[int]:
        n = embeddings.shape[0]
        if k >= n:
            return list(range(n))

        sims_to_query = _cosine_similarity(embeddings, query_vec)

        selected = []
        candidates = list(range(n))
        first = int(np.argmax(sims_to_query))
        selected.append(first)
        candidates.remove(first)

        while len(selected) < k and candidates:
            best_candidate = None
            best_score = -np.inf
            for candidate in candidates:
                relevance = sims_to_query[candidate]
                if selected:
                    selected_matrix = embeddings[selected]
                    diversity = _cosine_similarity(
                        selected_matrix, embeddings[candidate]
                    ).max()
                else:
                    diversity = 0.0
                score = lam * relevance - (1 - lam) * diversity
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            selected.append(best_candidate)  # type: ignore[arg-type]
            candidates.remove(best_candidate)  # type: ignore[arg-type]
        return selected

    def select(
        self,
        texts: list[str],
        task: str | None,
        keep_ratio: float = 0.4,
        lam: float = 0.5,
    ) -> tuple[list[int], list[float]]:
        embeddings = self.embed(texts)
        if embeddings.size == 0:
            return [], []

        if task:
            task_embedding = self.embed([task])[0]
        else:
            task_embedding = np.mean(embeddings, axis=0)

        k = max(1, int(len(texts) * keep_ratio))
        indices = self.mmr(embeddings, task_embedding, k=k, lam=lam)
        scores = _cosine_similarity(embeddings[indices], task_embedding).tolist()
        index_to_score = dict(zip(indices, scores, strict=False))
        ordered_indices = sorted(indices)
        ordered_scores = [index_to_score[idx] for idx in ordered_indices]
        return ordered_indices, ordered_scores
