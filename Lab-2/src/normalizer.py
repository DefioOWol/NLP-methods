import pymorphy3
from nltk.corpus import stopwords

MORPH = pymorphy3.MorphAnalyzer()


def load_stopwords() -> set[str]:
    try:
        return set(stopwords.words("russian")) - {"не", "нет", "но", "однако"}
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
    return load_stopwords()


STOPWORDS = load_stopwords()


def norm_token(token: str) -> str:
    token = token.lower()
    parsed = MORPH.parse(token)
    return parsed[0].normal_form


def normalize(tokens: list[str]) -> list[str]:
    normalized = []
    for token in tokens:
        if token.lower() in STOPWORDS:
            continue
        normalized.append(norm_token(token))
    return normalized
