from pathlib import Path
from typing import Self

import joblib
import numpy as np
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize as l2_normalize

from src.normalizer import normalize
from src.tokenizer import tokenize


def analyze_text(text: str) -> list[str]:
    return normalize(tokenize(text))


class LSAModel:
    def __init__(
        self,
        n_topic_components: int = 50,
        vector_size: int = 100,
        window: int = 5,
        epochs: int = 10,
    ):
        self._vectorizer = TfidfVectorizer(
            lowercase=False,
            analyzer=analyze_text,
            max_df=0.95,
            min_df=2,
            max_features=5000,
            sublinear_tf=True,
        )
        self._svd = TruncatedSVD(
            n_components=n_topic_components, random_state=42
        )

        self._vector_size = vector_size
        self._window = window
        self._epochs = epochs
        self._word2vec = None
        self._doc2vec = None

        self._texts = None
        self._topics_norm = None

    def fit(self, texts: list[str]) -> Self:
        self._texts = texts[:]

        tfidf = self._vectorizer.fit_transform(self._texts)
        topics = self._svd.fit_transform(tfidf)
        self._topics_norm = l2_normalize(topics)

        tokenized = [analyze_text(text) for text in self._texts]
        self._word2vec = Word2Vec(
            sentences=tokenized,
            vector_size=self._vector_size,
            window=self._window,
            epochs=self._epochs,
            min_count=2,
            workers=1,
            seed=42,
            sg=1,
        )

        tagged = [
            TaggedDocument(words=tokens, tags=[i])
            for i, tokens in enumerate(tokenized)
        ]
        self._doc2vec = Doc2Vec(
            documents=tagged,
            vector_size=self._vector_size,
            window=self._window,
            epochs=self._epochs,
            min_count=2,
            workers=1,
            seed=42,
            dm=1,
        )
    
        return self

    def top_tfidf(self, text: str, *, k: int = 5) -> list[tuple[str, float]]:
        terms = self._vectorizer.get_feature_names_out()
        vector = self._vectorizer.transform([text])
        row = vector.toarray().ravel()

        mask = np.flatnonzero(row)
        if mask.size == 0:
            return []

        top_idx = mask[np.argsort(row[mask], descending=True)[:k]]
        return [(terms[i], row[i]) for i in top_idx]

    def topic_terms(
        self, topic_id: int, *, k: int = 5
    ) -> list[tuple[str, float]]:
        terms = self._vectorizer.get_feature_names_out()
        weights = self._svd.components_[topic_id]
        top_idx = np.argsort(np.abs(weights), descending=True)[:k]
        return [(terms[i], weights[i]) for i in top_idx]

    def vectorize_topic(self, topic: str) -> np.ndarray:
        tfidf = self._vectorizer.transform([topic])
        return self._svd.transform(tfidf)[0]

    def steer_topic(
        self, topic_vector: np.ndarray, weights: dict[int, float]
    ) -> np.ndarray:
        steered = topic_vector.copy()
        for idx, weight in weights.items():
            steered[..., idx] += weight
        return steered

    def topic_similar(
        self,
        topic_vector: np.ndarray,
        *,
        k: int = 5,
        preview_length: int | None = None,
    ) -> list[tuple[str, float]]:
        query = l2_normalize(topic_vector.reshape(1, -1))
        scores = cosine_similarity(query, self._topics_norm).ravel()
        order = np.argsort(scores, descending=True)

        results = []
        for i in order:
            results.append(
                (self._preview(self._texts[i], preview_length), scores[i])
            )
            if len(results) >= k:
                break

        return results

    def similar_words(self, word: str, *, k: int = 5) -> list[tuple[str, float]]:
        return [
            (word, score)
            for word, score in self._word2vec.wv.most_similar(
                positive=[normalize(word)], topn=k
            )
        ]

    def vectorize_document(self, text: str) -> np.ndarray:
        tokens = analyze_text(text)
        return self._doc2vec.infer_vector(tokens, epochs=100)

    def similar_documents(
        self,
        text: str,
        *,
        k: int = 5,
        preview_length: int | None = None,
    ) -> list[tuple[str, float]]:
        vector = self.vectorize_document(text)
        return [
            (self._preview(self._texts[tag], preview_length), score)
            for tag, score in self._doc2vec.dv.most_similar(
                positive=[vector], topn=k
            )
        ]

    def pairwise_documents(self, texts: list[str]) -> np.ndarray:
        vectors = l2_normalize(
            np.vstack([self.vectorize_document(text) for text in texts])
        )
        return cosine_similarity(vectors)

    @staticmethod
    def _preview(text: str, length: int | None) -> str:
        text = text.replace("\n", " ").strip()
        if length and len(text) > length:
            return text[:length] + "..."
        return text

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return joblib.load(Path(path))
