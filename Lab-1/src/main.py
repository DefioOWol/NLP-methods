import argparse
from pathlib import Path

from normalizer import normalize
from sentiment import SentimentModel
from tokenizer import tokenize


MODEL_PATH = Path("models/model.joblib")


def analyze_text(model: SentimentModel, texts: list[str]):
    print()
    print("=" * 10)
    print("SENTIMENT")
    print("=" * 10)

    preds = model.predict(texts)

    for text, pred in zip(texts, preds, strict=True):
        print(text + " " * 10 + f"|    {pred.upper()}")

    print()


def show_preprocess_text(text: str):
    print()
    print("=" * 14)
    print("PREPROCESSING")
    print("=" * 14)

    print("\nИсходный текст:")
    print(text)

    tokens = tokenize(text)

    print("\nТокены:")
    print(tokens)

    print("\nНормализованные токены:")
    print(normalize(tokens))
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text sentiment analyzer")
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        help="Текст для анализа"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Путь к TXT-файлу"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Показать результат токенизации и нормализации для --text"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Необходимо указать --text или --file")

    if args.text and args.file:
        parser.error("Используйте только один из --text или --file")

    if args.text:
        text = args.text
    else:
        path = args.file
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path}")
        text = path.read_text(encoding="utf-8")

    if args.text and args.preprocess:
        show_preprocess_text(text)

    if not MODEL_PATH.exists():
        print("Модель не найдена. Выполните сперва: python -m src.train")
        return

    model = SentimentModel.load(MODEL_PATH)
    analyze_text(model, text.split("\n"))


if __name__ == "__main__":
    main()
