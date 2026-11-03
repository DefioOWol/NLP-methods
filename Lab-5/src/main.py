import argparse
from pathlib import Path

import numpy as np

from src.lsa import LSAModel, analyze_text

MODEL_PATH = Path("models/lsa.joblib")


def header(title: str):
    print("\n" + "-" * len(title))
    print(title)
    print("-" * len(title))


def print_topic_similar(
    model: LSAModel, topic_vector: np.ndarray, *, k: int = 5
):
    header("ДОКУМЕНТЫ, БЛИЗКИЕ К ТЕМЕ")
    for text, score in model.topic_similar(
        topic_vector, k=k, preview_length=100
    ):
        print(f"  [{score:.4f}] {text}")


def print_similar_words(model: LSAModel, word: str, *, k: int = 5):
    header(f"БЛИЗКИЕ СЛОВА (Word2Vec) ДЛЯ '{word}'")
    for w, score in model.similar_words(word, k=k):
        print(f"  {w}: {score:.4f}")


def print_similar_documents(model: LSAModel, text: str, *, k: int = 5):
    header("ПОХОЖИЕ ДОКУМЕНТЫ (Doc2Vec)")
    for doc, score in model.similar_documents(text, k=k, preview_length=100):
        print(f"  [{score:.4f}] {doc}")


def print_pairwise(model: LSAModel, texts: list[str]):
    header("ПОПАРНАЯ БЛИЗОСТЬ ДОКУМЕНТОВ (Cosine Similarity)")
    n = len(texts)

    header_line = " " * 8 + "".join(f"doc_{j:<5}" for j in range(n))
    print(header_line)

    matrix = model.pairwise_documents(texts)
    for i in range(n):
        row = f"doc_{i:<4} " + "".join(
            f"{matrix[i, j]:<8.4f}" for j in range(n)
        )
        print(row)


def print_top_tfidf(model: LSAModel, text: str, k: int = 5):
    header("3. КЛЮЧЕВЫЕ СЛОВА (TF-IDF)")
    for word, score in model.top_tfidf(text, k=k):
        print(f"  {word}: {score:.4f}")


def print_topic_similar(
    model: LSAModel,
    text: str,
    steer_weights: dict[int, float] | None,
    *,
    k: int = 5,
):
    header("4. БЛИЗКИЕ ДОКУМЕНТЫ ПО ТЕМЕ (Cosine Similarity)")
    topic_vector = model.vectorize_topic(text)
    if steer_weights:
        topic_vector = model.steer_topic(topic_vector, steer_weights)

    neighbors = model.topic_similar(topic_vector, k=k, preview_length=100)
    if not neighbors:
        print("Соседи не найдены")
    else:
        for i, (doc, score) in enumerate(neighbors, start=1):
            print(f"  {i}. [{score:.4f}] {doc}")


def print_classification(model: LSAModel, text: str):
    header("5. АНАЛИЗ ТОНАЛЬНОСТИ (TextCNN)")
    label, probs = model.classify_corpus(text)
    prob_str = ", ".join(f"{cls}: {p:.4f}" for cls, p in probs.items())
    print(f"  Класс тональности: {label.upper()}")
    print(f"  Распределение вероятностей: {prob_str}")


def print_generation(model: LSAModel, text: str):
    summary = model.extract_semantic_summary(text)
    extraction = summary["extraction"]

    header("6. СИНТАКСИЧЕСКИЕ ВЗАИМОСВЯЗИ И ЧАСТИ РЕЧИ (spaCy)")
    if extraction.relations:
        print("  Извлеченные взаимосвязи (SVO):")
        for r in extraction.relations:
            print(
                f"    - Субъект: {r.subject:<12} | Действие: {r.action:<14}"
                f" | Объект: {r.target}"
            )
    else:
        print("  Извлеченные взаимосвязи: отсутствуют явные пары")

    if extraction.key_phrases:
        print(
            "  Ключевые синтаксические фразы (POS-patterns): "
            + ", ".join(extraction.key_phrases)
        )

    header("7. ГЕНЕРАЦИЯ ОСМЫСЛЕННОГО ТЕКСТА (Seq2Seq LSTM + Templates)")
    commands_str = " -> ".join(summary["commands"])
    print(f"  Последовательность шаблонов: {commands_str}")

    print("\n  Сгенерированный текст:")
    printed_len = 0
    for part in model.generate(text).split():
        if printed_len == 0:
            print(end="  ")

        print(part, end=" ")
        printed_len += len(part)

        if printed_len >= 100:
            printed_len = 0
            print()

    if printed_len:
        print()


def print_analyze(
    model: LSAModel,
    text: str,
    steer_weights: dict[int, float] | None,
):
    header("1. ИСХОДНЫЙ ТЕКСТ")
    print(f"  {text}")

    header("2. ПРЕДОБРАБОТКА (ЛЕММАТИЗАЦИЯ)")
    lemmas = analyze_text(text)
    print(f"  {' '.join(lemmas)}")

    print_top_tfidf(model, text)
    print_topic_similar(model, text, steer_weights)
    print_classification(model, text)
    print_generation(model, text)

    print("\n" + "=" * 21)
    print("ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ")
    print("=" * 21)
    for lemma in lemmas:
        print_similar_words(model, lemma)
    print_similar_documents(model, text)


def parse_steer(value: str) -> dict[int, float]:
    parts = [part for p in value.split(",") if (part := p.strip())]
    if not parts:
        raise argparse.ArgumentTypeError("Пустой список весов")

    weights = {}
    for part in parts:
        if ":" not in part:
            raise argparse.ArgumentTypeError(f"Некорректный формат: {part}")

        idx, weight = part.split(":", 1)

        try:
            idx, weight = int(idx.strip()), float(weight.strip())
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"Не числа: {part}") from e

        if idx in weights:
            raise argparse.ArgumentTypeError(f"Повтор id: {idx}")

        weights[idx] = weight

    return weights


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LSA semantic analyzer")
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        help="Текст для анализа",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Путь к TXT-файлу",
    )
    parser.add_argument(
        "--steer",
        type=parse_steer,
        help=(
            "Сдвиг тем запроса, формат id:weight"
            " через запятую (например 0:1.5,3:-0.8)"
        ),
    )
    return parser


def load_texts(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> list[str]:
    if not args.text and not args.file:
        parser.error("Необходимо указать --text или --file")

    if args.text and args.file:
        parser.error("Используйте только --text или --file")

    if args.text:
        return [args.text]

    path = args.file
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")

    return [
        line
        for l in path.read_text(encoding="utf-8").splitlines()
        if (line := l.strip())
    ]


def main():
    parser = build_parser()
    args = parser.parse_args()

    texts = load_texts(args, parser)
    model = LSAModel.load(MODEL_PATH)

    if args.steer:
        for idx in args.steer:
            if idx < 0 or idx >= len(model._svd.components_):
                parser.error(
                    f"Индекс темы {idx} вне диапазона"
                    f" [0, {len(model._svd.components_)})"
                )

    for text in texts:
        print_analyze(model, text, args.steer)
        print()

    if len(texts) >= 2:
        print_pairwise(model, texts)
        print()


if __name__ == "__main__":
    main()
