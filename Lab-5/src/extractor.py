import re
from dataclasses import dataclass

import spacy
from spacy.tokens import Doc, Token
from spacy.matcher import Matcher

NLP = spacy.load("ru_core_news_sm")

MONTHS = (
    "январ[яеь]|феврал[яеь]|март[ае]?|апрел[яеь]|ма[яйе]|"
    "июн[яеь]|июл[яеь]|август[ае]?|сентябр[яеь]|"
    "октябр[яеь]|ноябр[яеь]|декабр[яеь]"
)
MONTH_WORDS = (
    "январь|февраль|март|апрель|май|июнь|"
    "июль|август|сентябрь|октябрь|ноябрь|декабрь"
)
WEEKDAYS = (
    "понедельник(?:а|у|ом)?|вторник(?:а|у|ом)?|"
    "сред[аеуы]|четверг(?:а|у|ом)?|"
    "пятниц[аеуы]|суббот[аеуы]|воскресень[еяю]"
)
UNITS = (
    r"штук[аи]?|шт\.?|кг|килограмм(?:а|ов)?|грамм(?:а|ов)?|"
    r"л|литр(?:а|ов)?|мл|м|км|см|мм|метр(?:а|ов)?|"
    r"человек(?:а)?|чел\.?|раз(?:а)?|"
    r"комнат[аы]?|этаж(?:а|ей)?|мест(?:а|о)?"
)
NUM_WORD = r"\d+(?:[.,]\d+)?"
MONEY_UNIT = (
    r"(?:тыс\.?|млн\.?|млрд\.?|миллион(?:а|ов)?|миллиард(?:а|ов)?|"
    r"тысяч[аи]?)?\s?"
    r"(?:руб(?:лей|ля|ль)?\.?|р\.|₽|евро|долларов|доллара|\$|usd|eur)"
)
INTERVAL_FROM = r"(?:с|от|между)"
INTERVAL_TO = r"(?:по|до|и)"
INTERVAL_POINT = (
    r"(?:\d{1,2}[:.]\d{2}|\d{1,2}\s(?:"
    + MONTHS
    + r")(?:\s(?:19|20)\d{2})?|\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?)"
)

FACT_NAMES = {
    "money": "суммы",
    "quantity": "количества",
    "date": "даты",
    "time": "время",
    "interval": "интервалы",
    "duration": "сроки",
    "weekday": "дни недели",
    "month": "месяцы",
    "percent": "проценты",
    "phone": "телефоны",
    "address": "адреса",
    "geo": "геометки",
    "url": "ссылки",
    "email": "почта",
    "number": "номера",
}

FACT_PATTERNS = {
    "money": re.compile(
        rf"\b{NUM_WORD}(?:\s?\d{{3}})*\s?{MONEY_UNIT}\b",
        re.IGNORECASE,
    ),
    "quantity": re.compile(
        rf"\b{NUM_WORD}\s(?:{UNITS})\b",
        re.IGNORECASE,
    ),
    "date": re.compile(
        r"\b(?:[0123]?\d[./][01]?\d[./](?:\d\d)?\d\d|"
        rf"[0123]?\d\s(?:{MONTHS})(?:\s(?:19|20)\d{{2}}(?:\s?(?:г\.|года))?)?)\b",
        re.IGNORECASE,
    ),
    "time": re.compile(
        r"\b(?:[01]?\d|2[0-3])[:.]\s?[0-5]\d(?:\s?(?:час(?:а|ов)?|ч\.?))?\b"
        r"|\b(?:в\s)?(?:[01]?\d|2[0-3])\sчас(?:а|ов)?\b",
        re.IGNORECASE,
    ),
    "interval": re.compile(
        rf"\b{INTERVAL_FROM}\s{INTERVAL_POINT}\s{INTERVAL_TO}\s{INTERVAL_POINT}\b",
        re.IGNORECASE,
    ),
    "duration": re.compile(
        rf"\b{NUM_WORD}\s(?:минут[уы]?|мин\.?|час(?:а|ов)?|ч\.?|"
        r"дн(?:я|ей)|день|сут(?:ки|ок)|недел[июье]|мес(?:яц(?:а|ев)?|\.)|"
        r"год(?:а)?|лет)\b",
        re.IGNORECASE,
    ),
    "weekday": re.compile(rf"\b(?:{WEEKDAYS})\b", re.IGNORECASE),
    "month": re.compile(rf"\b(?:{MONTH_WORDS})\b", re.IGNORECASE),
    "percent": re.compile(rf"\b{NUM_WORD}\s?%"),
    "phone": re.compile(
        r"(?:\+7|\b8)[\s(-]*\d{3}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}\b"
    ),
    "address": re.compile(
        r"\b(?:ул\.|улиц[аеуы]|пр\.|проспект[ае]?|пер\.|переулок)\s+"
        r"[А-ЯЁа-яё]{2,}(?:\s*,\s*(?:д\.|дом)?\s*\d+[А-Яа-я/-]*)?"
        r"(?:\s*,?\s*(?:корп\.|к\.|кв\.)\s*\d+)?",
        re.IGNORECASE,
    ),
    "geo": re.compile(
        r"\b(?:г\.|город)\s?[А-ЯЁ][а-яё-]{2,}"
        r"|\b[А-ЯЁ][а-яё-]{2,}(?:ская|ский|ское)\s(?:область|край|район)\b"
        r"|\b(?:-?\d{1,3}\.\d{3,})\s*[,;]\s*(?:-?\d{1,3}\.\d{3,})\b"
    ),
    "url": re.compile(
        r"\b(?:https?://|www\.)[^\s<>\"']+",
        re.IGNORECASE,
    ),
    "email": re.compile(
        r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b",
        re.IGNORECASE,
    ),
    "number": re.compile(
        r"№\s?[\w-]{1,20}\b"
        r"|\bномер\s?[\w-]{1,20}\b"
        r"|\b(?:ИНН|КПП|ОГРН|БИК|СНИЛС)\s?\d[\d\s-]{8,}\b"
        r"|\b(?:паспорт|серия)\s?\d{2}\s?\d{2}\s?\d{6}\b",
        re.IGNORECASE,
    ),
}

_DROP_IF_INSIDE = {
    "month": ("date", "interval"),
    "weekday": ("interval",),
    "time": ("interval",),
    "date": ("interval",),
    "quantity": ("money", "duration", "percent"),
    "number": ("money", "phone", "date"),
}


def _is_inside(start: int, end: int, left: int, right: int) -> bool:
    return start >= left and end <= right


def _add_fact(facts: dict[str, list[str]], kind: str, value: str):
    value = " ".join(value.split()).rstrip(".,;:)")
    if not value:
        return
    bucket = facts.setdefault(kind, [])
    if value not in bucket:
        bucket.append(value)


def extract_facts(text: str) -> dict[str, list[str]]:
    hits = []
    for kind, pattern in FACT_PATTERNS.items():
        for match in pattern.finditer(text):
            value = " ".join(match.group().split()).rstrip(".,;:)")
            if not value:
                continue
            hits.append((kind, value, match.start(), match.end()))

    facts = {}
    for kind, value, start, end in hits:
        parents = _DROP_IF_INSIDE.get(kind, ())
        if any(
            other in parents and _is_inside(start, end, left, right)
            for other, _, left, right in hits
        ):
            continue
        _add_fact(facts, kind, value)
    return facts


_COUNT_STOP = {
    "год", "года", "лет", "день", "дня", "дней", "час", "часа", "часов",
    "минута", "минуты", "месяц", "месяца", "неделя", "недели", "рубль",
    "рубля", "рублей", "доллар", "евро", "процент",
}
_GEO_LABELS = {"GPE", "LOC", "FAC"}


def enrich_facts(doc: Doc, facts: dict[str, list[str]]) -> dict[str, list[str]]:
    for ent in doc.ents:
        if ent.label_ in _GEO_LABELS:
            _add_fact(facts, "geo", ent.text)
        elif ent.label_ == "DATE":
            _add_fact(facts, "date", ent.text)
        elif ent.label_ == "TIME":
            _add_fact(facts, "time", ent.text)
        elif ent.label_ in ("MONEY", "PERCENT"):
            _add_fact(facts, ent.label_.lower(), ent.text)

    for token in doc:
        if token.dep_ != "nummod" or token.head.pos_ != "NOUN":
            continue

        head = token.head.text.lower().rstrip(".")
        lemma = token.head.lemma_.lower().rstrip(".")
        if head in _COUNT_STOP or lemma in _COUNT_STOP:
            continue

        phrase = f"{token.text} {token.head.text}"
        blob = " ".join(facts.get("money", []) + facts.get("address", []))
        if phrase in blob or token.text in blob.split():
            continue

        _add_fact(facts, "quantity", phrase)

    return facts


def facts_summary(facts: dict[str, list[str]]) -> str:
    parts = [
        f"{FACT_NAMES[kind]}: {', '.join(values)}"
        for kind, values in facts.items()
    ]
    return "; ".join(parts)


@dataclass
class Relation:
    subject: str
    action: str
    target: str
    relation_type: str = "svo"

    def as_phrase(self) -> str:
        parts = [self.subject, self.action]
        if self.target and self.target != "-":
            parts.append(self.target)
        return " ".join(p for p in parts if p)


@dataclass
class ExtractResult:
    relations: list[Relation]
    pos_tags: list[tuple[str, str]]
    key_phrases: list[str]
    facts: dict[str, list[str]]

    def relation_summary(self, *, k: int = 3) -> str:
        if not self.relations:
            if self.key_phrases:
                return "; ".join(self.key_phrases[:k])
            return ""
        return "; ".join(r.as_phrase() for r in self.relations[:k])


def _arg(verb: Token, deps: tuple[str, ...]):
    return next((t for t in verb.children if t.dep_ in deps), None)


def extract_with_spacy(doc: Doc) -> list[Relation]:
    relations = []
    seen = set()

    for sent in doc.sents:
        verbs = [
            t for t in sent
            if t.pos_ == "VERB" and t.dep_ in ("ROOT", "conj", "ccomp", "xcomp")
        ]
        if not verbs:
            verbs = [t for t in sent if t.pos_ == "VERB"][:1]

        for verb in verbs:
            subj = next((t for t in verb.children if "subj" in t.dep_), None)
    
            if subj is None:
                subj = _arg(verb.head, ("nsubj", "nsubj:pass"))
                if subj is None:
                    continue

            obj = _arg(verb, ("obj", "iobj"))
            obj = obj.text if obj else ""

            triple = (subj.text.lower(), verb.text.lower(), obj.lower())
            if triple in seen:
                continue
            seen.add(triple)

            relations.append(
                Relation(
                    subject=subj.text,
                    action=verb.text,
                    target=obj,
                    relation_type="syntactic_svo",
                )
            )

    matcher = Matcher(doc.vocab)
    matcher.add(
        "ACTOR_ACTION",
        [[{"POS": {"IN": ["NOUN", "PROPN", "PRON"]}}, {"POS": "VERB"}]],
    )

    for _, start, end in matcher(doc):
        span = doc[start:end]
        s, a = span[0].text, span[1].text
        triple = (s.lower(), a.lower(), "")

        if triple not in seen:
            seen.add(triple)
            relations.append(
                Relation(
                    subject=s, action=a, target="", relation_type="pos_pattern"
                )
            )

    return relations


def extract_relations(text: str) -> ExtractResult:
    doc = NLP(text)

    key_phrases = []
    for token in doc:
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            key_phrases.append(f"{token.text} {token.head.text}")

    return ExtractResult(
        relations=extract_with_spacy(doc),
        pos_tags=[(t.text, t.pos_) for t in doc],
        key_phrases=key_phrases,
        facts=enrich_facts(doc, extract_facts(text)),
    )
