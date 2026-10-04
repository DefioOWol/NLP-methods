import re

TOKEN_PATTERN = re.compile(
    r"[0-9]+(?:[.,][0-9]+)*|[а-яёa-z]+(?:[.-_][а-яёa-z]+)*", re.IGNORECASE
)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)
