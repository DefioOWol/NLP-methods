from pathlib import Path

import pandas as pd

from src.lsa import LSAModel

DATASET_PATH = Path("data") / "dataset.csv"
TEXT_COL = "text"

MODEL_PATH = Path("models") / "lsa.joblib"
N_COMPONENTS = 50
MAX_DOCS = 20000

TOPIC_LIMIT = 3
TOP_TERMS = 10


def main():
    print("Loading dataset...", end=" ")

    df = pd.read_csv(DATASET_PATH, usecols=[TEXT_COL])
    df = df.dropna(subset=[TEXT_COL])
    df[TEXT_COL] = df[TEXT_COL].str.strip()
    df = df[df[TEXT_COL].ne("")]

    if len(df) > MAX_DOCS:
        df = df.sample(n=MAX_DOCS, random_state=42)

    texts = df[TEXT_COL].tolist()

    print("done")
    print(f"Dataset size: {len(texts)}")

    print("Training LSA...", end=" ")

    model = LSAModel(N_COMPONENTS)
    model.fit(texts)

    print("done")

    explained = float(model._svd.explained_variance_ratio_.sum())
    print(f"n_components: {len(model._svd.components_)}")
    print(f"explained variance: {explained:.3f}\n")

    for topic_id in range(min(len(model._svd.components_), TOPIC_LIMIT)):
        terms = model.topic_terms(topic_id, k=TOP_TERMS)
        formatted = ", ".join(
            f"{term} ({weight:.3f})" for term, weight in terms
        )
        print(f"Topic {topic_id}: {formatted}")

    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
