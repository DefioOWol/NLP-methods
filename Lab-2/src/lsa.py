from pathlib import Path
from typing import Self

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as l2_normalize

from src.normalizer import normalize
from src.tokenizer import tokenize


def analyze_text(text: str) -> list[str]:
    return normalize(tokenize(text))


class LSAModel:
    def __init__(self, n_components: int = 50):
        self._vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer=analyze_text,
            max_df=0.95,
            min_df=2,
            max_features=5000,
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._texts = None
        self._topics_norm = None

    def fit(self, texts: list[str]) -> Self:
        self._texts = texts[:]
        tfidf = self._vectorizer.fit_transform(self._texts)
        topics = self._svd.fit_transform(tfidf)
        self._topics_norm = l2_normalize(topics)
        return self

    def transform_topics(self, texts: list[str]) -> np.ndarray:
        tfidf = self._vectorizer.transform(texts)
        return self._svd.transform(tfidf)

    def topic_terms(
        self, topic_id: int, *, k: int = 10
    ) -> list[tuple[str, float]]:
        terms = self._vectorizer.get_feature_names_out()
        weights = self._svd.components_[topic_id]
        top_idx = np.argsort(np.abs(weights), descending=True)[:k]
        return [(terms[i], weights[i]) for i in top_idx]

    def top_tfidf(self, text: str, *, k: int = 10) -> list[tuple[str, float]]:
        terms = self._vectorizer.get_feature_names_out()
        vector = self._vectorizer.transform([text])
        row = vector.toarray().ravel()

        mask = np.flatnonzero(row)
        if mask.size == 0:
            return []

        top_idx = mask[np.argsort(row[mask], descending=True)[:k]]
        return [(terms[i], row[i]) for i in top_idx]

    def steer(
        self, topic_vector: np.ndarray, weights: dict[int, float]
    ) -> np.ndarray:
        steered = np.asarray(topic_vector, dtype=np.float32, copy=True)
        for idx, weight in weights.items():
            steered[..., idx] += weight
        return steered

    def similar(
        self,
        topic_vector: np.ndarray,
        *,
        k: int = 5,
        exclude_text: str | None = None,
        preview_length: int | None = None,
    ) -> list[tuple[str, float]]:
        query = l2_normalize(
            np.asarray(topic_vector, dtype=np.float32).reshape(1, -1)
        )
        scores = cosine_similarity(query, self._topics_norm).ravel()
        order = np.argsort(scores, descending=True)

        results = []
        for i in order:
            text = self._texts[i]
            if exclude_text is not None and text == exclude_text:
                continue

            results.append((self._preview(text, preview_length), scores[i]))
            if len(results) >= k:
                break

        return results

    def _preview(self, text: str, length: int | None) -> str:
        text = text.replace("\n", " ").strip()
        if length and len(text) > length:
            return text[:length] + "..."
        return text

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "vectorizer": self._vectorizer,
                "svd": self._svd,
                "texts": self._texts,
                "topics_norm": self._topics_norm,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        data = joblib.load(Path(path))
        svd = data["svd"]

        model = cls(n_components=svd.n_features_in_)
        model._vectorizer = data["vectorizer"]
        model._svd = svd
        model._texts = data["texts"]
        model._topics_norm = data["topics_norm"]

        return model
