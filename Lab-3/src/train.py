from pathlib import Path

import pandas as pd

from src.lsa import LSAModel

DATASET_PATH = Path("data") / "dataset.csv"
DATASET_TEXT_COL = "text"
DATASET_MAX_DOCS = 20000

LSA_PATH = Path("models") / "lsa.joblib"
LSA_N_TOPIC_COMPONENTS = 50
LSA_VECTOR_SIZE = 100
LSA_WINDOW = 5
LSA_EPOCHS = 10

TOPIC_LIMIT = 5
TOP_TOPIC_TERMS = 5

WORD_SAMPLES = ("фильм", "товар", "плохой")
TOP_SIMILAR_WORDS = 5


def load_texts() -> list[str]:
    df = pd.read_csv(DATASET_PATH, usecols=[DATASET_TEXT_COL])
    df = df.dropna(subset=[DATASET_TEXT_COL])
    df[DATASET_TEXT_COL] = df[DATASET_TEXT_COL].astype(str).str.strip()
    df = df[df[DATASET_TEXT_COL].ne("")]

    if len(df) > DATASET_MAX_DOCS:
        df = df.sample(n=DATASET_MAX_DOCS, random_state=42)

    return df[DATASET_TEXT_COL].tolist()


def main():
    print("Loading dataset...", end=" ")
    texts = load_texts()
    print("done")
    print(f"Dataset size: {len(texts)}\n")

    print("Training LSA...", end=" ")
    lsa = LSAModel(
        n_topic_components=LSA_N_TOPIC_COMPONENTS,
        vector_size=LSA_VECTOR_SIZE,
        window=LSA_WINDOW,
        epochs=LSA_EPOCHS,
    )
    lsa.fit(texts)
    print("done\n")

    explained = float(lsa._svd.explained_variance_ratio_.sum())
    print(f"n_topic_components: {len(lsa._svd.components_)}")
    print(f"explained variance: {explained:.3f}\n")

    for topic_id in range(min(len(lsa._svd.components_), TOPIC_LIMIT)):
        terms = lsa.topic_terms(topic_id, k=TOP_TOPIC_TERMS)
        formatted = ", ".join(
            f"{term} ({weight:.3f})" for term, weight in terms
        )
        print(f"Topic {topic_id}: {formatted}")

    print(f"\nvector size: {LSA_VECTOR_SIZE}")
    print(f"vocabulary: {len(lsa._word2vec.wv)}\n")

    for sample in WORD_SAMPLES:
        neighbors = lsa.similar_words(sample, k=TOP_SIMILAR_WORDS)
        if not neighbors:
            print(f"Word {sample}: not in vocabulary")
            continue
        formatted = ", ".join(
            f"{word} ({score:.3f})" for word, score in neighbors
        )
        print(f"Word '{sample}': {formatted}")

    lsa.save(LSA_PATH)
    print(f"\nLSA saved to: {LSA_PATH}")


if __name__ == "__main__":
    main()
