from pathlib import Path
from typing import Self

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from normalizer import normalize
from tokenizer import tokenize


def analyze_text(text: str) -> list[str]:
    return normalize(tokenize(text))


class SentimentModel:
    def __init__(self):
        self._pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    analyzer=analyze_text,
                    max_df=0.95,
                    min_df=1,
                    max_features=5000,
                    sublinear_tf=True,
                ),
            ),
            ("cls", MultinomialNB()),
        ])

    def fit(self, texts: list[str], labels: list[str]) -> Self:
        self._pipeline.fit(texts, labels)
        return self

    def predict(self, text: str | list[str]) -> str | list[str]:
        single = isinstance(text, str)
        pred = self._pipeline.predict([text] if single else text)
        return pred[0] if single else pred.tolist()

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        model = cls()
        model._pipeline = joblib.load(Path(path))
        return model
