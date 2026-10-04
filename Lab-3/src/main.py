import argparse
from pathlib import Path

import numpy as np

from src.lsa import LSAModel
from src.normalizer import normalize
from src.tokenizer import tokenize

MODEL_PATH = Path("models") / "lsa.joblib"

TOP_TERMS = 5
TOP_TOPICS = 5
TOP_SIMILAR = 5
WORDS_LIMIT = 5

PREVIEW_LENGTH = 100


def header(title: str):
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def format_vector(vector: np.ndarray) -> str:
    return np.array2string(
        vector, separator=", ", precision=4, max_line_width=80
    )


def format_matrix(matrix: np.ndarray) -> str:
    result = ""
    for row in matrix:
        result += format_vector(row) + "\n"
    return result[:-1]


def format_terms(pairs: list[tuple[str, float]]) -> str:
    if not pairs:
        return "(пусто)"
    return ", ".join(f"{term}: {weight:.4f}" for term, weight in pairs)


def show_preprocess_text(text: str) -> list[str]:
    header("PREPROCESSING")

    print("\nИсходный текст:")
    print(text)

    tokens = tokenize(text)
    print("\nТокены:")
    print(tokens)

    lemmas = normalize(tokens)
    print("\nНормализованные токены:")
    print(lemmas)
    print()

    return lemmas


def analyze_text(
    model: LSAModel,
    text: str,
    steer_weights: dict[int, float] | None,
):
    lemmas = show_preprocess_text(text)
    if not lemmas:
        print("После нормализации текст пуст, разбор пропущен")
        return

    header("TF-IDF")
    print(format_terms(model.top_tfidf(text, k=TOP_TERMS)))

    header("TOPIC VECTOR")
    topic_vector = model.vectorize_topic(text)
    print(format_vector(topic_vector))

    ranked_topics = sorted(
        enumerate(topic_vector),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:TOP_TOPICS]
    print("\nТоп тем по |весу|:")
    for topic_id, weight in ranked_topics:
        print(f"   {topic_id}: {weight:.4f}")

    header("TOPICS")
    for topic_id, weight in ranked_topics:
        print(f"\nТема {topic_id} ({weight:.4f}):")
        print("    " + format_terms(model.topic_terms(topic_id, k=TOP_TERMS)))

    query_vector = topic_vector
    if steer_weights:
        query_vector = model.steer_topic(topic_vector, steer_weights)
        shifts = ", ".join(
            f"{idx} += {weight}" for idx, weight in steer_weights.items()
        )

        header("STEERING")
        print(f"Сдвиги: {shifts}")
        print("\nДо:")
        print(format_vector(topic_vector))
        print("\nПосле:")
        print(format_vector(query_vector))

    header("TOPIC SIMILAR")
    neighbors = model.topic_similar(
        query_vector, k=TOP_SIMILAR, preview_length=PREVIEW_LENGTH
    )
    if not neighbors:
        print("Соседи не найдены")
        return
    for i, (doc, score) in enumerate(neighbors, start=1):
        print(f"{i}. [{score:.4f}] {doc}")

    header("WORD2VEC")
    shown, seen = 0, set()
    for lemma in lemmas:
        if lemma in seen:
            continue
        seen.add(lemma)

        neighbors = model.similar_words(lemma, k=TOP_TERMS)
        if not neighbors:
            continue

        print(f"\n{lemma}:")
        print("    " + format_terms(neighbors))

        shown += 1
        if shown >= WORDS_LIMIT:
            break
    if shown == 0:
        print("Нет известных слов")

    header("DOC2VEC SIMILAR")
    neighbors = model.similar_documents(
        text,
        k=TOP_SIMILAR,
        preview_length=PREVIEW_LENGTH,
    )
    for i, (doc, score) in enumerate(neighbors, start=1):
        print(f"{i}. [{score:.4f}] {doc}")


def print_pairwise(model: LSAModel, texts: list[str]):
    matrix = model.pairwise_documents(texts)
    header("DOC2VEC PAIRWISE")
    print(format_matrix(matrix))


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
        analyze_text(model, text, args.steer)
        print()

    if len(texts) >= 2:
        print_pairwise(model, texts)
        print()


if __name__ == "__main__":
    main()
