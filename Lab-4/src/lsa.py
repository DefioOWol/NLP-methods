from collections import defaultdict
from pathlib import Path
from typing import Self

import joblib
import numpy as np
import torch
from gensim.models import Doc2Vec, Word2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize as l2_normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.normalizer import normalize
from src.tokenizer import tokenize

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unknown>"


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        kernel_sizes: tuple[int, ...],
        num_filters: int,
        num_classes: int,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=padding_idx
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=k)
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).transpose(1, 2)

        pooled = []
        for conv in self.convs:
            out = torch.relu(conv(embedded))
            pooled.append(torch.amax(out, dim=2))

        features = self.dropout(torch.cat(pooled, dim=1))
        out = self.fc(features)

        return out


def analyze_text(text: str) -> list[str]:
    return normalize(tokenize(text))


class LSAModel:
    def __init__(
        self,
        n_topic_components: int = 50,
        vector_size: int = 100,
        window: int = 5,
        epochs: int = 10,
        cnn_max_len: int = 100,
        cnn_filters: int = 128,
        cnn_kernel_sizes: tuple[int, ...] = (2, 3, 4),
        cnn_dropout: float = 0.3,
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

        self._cnn_max_len = cnn_max_len
        self._cnn_filters = cnn_filters
        self._cnn_kernel_sizes = cnn_kernel_sizes
        self._cnn_dropout = cnn_dropout
        self._cnn = None
        self._cnn_word2idx = {}
        self._cnn_label_encoder = LabelEncoder()

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

    def fit_cnn(
        self,
        texts: list[str],
        labels: list[str],
        *,
        epochs: int = 10,
        batch_size: int = 1024,
        lr: float = 1e-3,
    ) -> Self:
        tokenized = [analyze_text(text) for text in texts]
        self._cnn_word2idx = self._build_cnn_vocab(tokenized)
        encoded = self._encode_texts(tokenized)

        y = self._cnn_label_encoder.fit_transform(labels)

        self._cnn = TextCNN(
            vocab_size=len(self._cnn_word2idx),
            embed_dim=self._vector_size,
            kernel_sizes=self._cnn_kernel_sizes,
            num_filters=self._cnn_filters,
            num_classes=len(self._cnn_label_encoder.classes_),
            dropout=self._cnn_dropout,
            padding_idx=self._cnn_word2idx[PAD_TOKEN],
        )
        self._init_cnn_embeddings()

        dataset = TensorDataset(encoded, torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self._cnn.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self._cnn.train()
        for _ in range(epochs):
            for ids, labels in loader:
                optimizer.zero_grad()
                logits = self._cnn(ids)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

        self._cnn.eval()
        return self

    @torch.inference_mode()
    def classify_corpus(
        self, texts: str | list[str]
    ) -> tuple[str, dict[str, float]] | list[tuple[str, dict[str, float]]]:
        single = isinstance(texts, str)
        docs = [texts] if single else texts
        encoded = self._encode_texts([analyze_text(doc) for doc in docs])

        logits = self._cnn(encoded)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        classes = [str(c) for c in self._cnn_label_encoder.classes_]
        results = []
        for i, pred_id in enumerate(probs.argmax(axis=1)):
            dist = {lbl: float(probs[i, j]) for j, lbl in enumerate(classes)}
            results.append((classes[pred_id], dist))

        return results[0] if single else results

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
                (self.preview(self._texts[i], preview_length), scores[i])
            )
            if len(results) >= k:
                break

        return results

    def similar_words(self, word: str, *, k: int = 5) -> list[tuple[str, float]]:
        lemma = normalize(word)
        if lemma not in self._word2vec.wv:
            return []
        return [
            (word, score)
            for word, score in self._word2vec.wv.most_similar(
                positive=[lemma], topn=k
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
            (self.preview(self._texts[tag], preview_length), score)
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
    def preview(text: str, length: int | None) -> str:
        text = text.replace("\n", " ").strip()
        if length and len(text) > length:
            return text[:length] + "..."
        return text

    @staticmethod
    def _build_cnn_vocab(tokenized: list[list[str]]) -> dict[str, int]:
        counts = defaultdict(int)
        for tokens in tokenized:
            for token in tokens:
                counts[token] += 1

        word2idx = {PAD_TOKEN: 0, UNKNOWN_TOKEN: 1}
        for token, count in sorted(counts.items()):
            if count >= 2:
                word2idx[token] = len(word2idx)

        return word2idx

    def _encode_texts(self, tokenized: list[list[str]]) -> torch.Tensor:
        pad = self._cnn_word2idx[PAD_TOKEN]
        unknown = self._cnn_word2idx[UNKNOWN_TOKEN]

        rows = np.full((len(tokenized), self._cnn_max_len), pad, dtype=np.int64)

        for i, tokens in enumerate(tokenized):
            ids = [
                self._cnn_word2idx.get(token, unknown)
                for token in tokens[:self._cnn_max_len]
            ]
            if not ids:
                ids = [unknown]

            rows[i, :len(ids)] = ids

        return torch.from_numpy(rows)

    def _init_cnn_embeddings(self):
        weights = self._cnn.embedding.weight.data
        for word, idx in self._cnn_word2idx.items():
            if word in self._word2vec.wv:
                weights[idx] = torch.tensor(
                    self._word2vec.wv[word], dtype=weights.dtype
                )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return joblib.load(Path(path))
