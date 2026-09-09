from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from sentiment import SentimentModel

DATASET_PATH = Path("data/dataset.csv")

TEXT_COL = "text"
LABEL_COL = "label"

LABEL_MAP = {0: "neutral", 1: "positive", 2: "negative"}

MODEL_PATH = Path("models/model.joblib")


def main():
    print("Loading dataset...", end=" ")

    df = pd.read_csv(DATASET_PATH, usecols=[TEXT_COL, LABEL_COL])
    df = df.dropna(subset=[TEXT_COL, LABEL_COL])
    df[TEXT_COL] = df[TEXT_COL].str.strip()
    df = df[df[TEXT_COL].ne("")]

    X, y = df[TEXT_COL], df[LABEL_COL].map(LABEL_MAP)

    print("done")
    print(f"Dataset size: {len(df)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("Training model...", end=" ")

    model = SentimentModel()
    model.fit(X_train.tolist(), y_train.tolist())

    print("done")
    print("Evaluation...")

    preds = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds, zero_division=0))

    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
