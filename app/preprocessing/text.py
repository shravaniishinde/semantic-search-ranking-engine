"""Deterministic text preprocessing.

Two levels on purpose:
* ``clean_text`` is light normalisation that keeps casing, punctuation and stopwords, because
  transformer embeddings and spaCy NER rely on them.
* ``tokenize`` is the lexical view used for BM25 (in-memory backend) and ranking features:
  lowercased tokens, a small stopword list and a minimal plural-stripping rule. Elasticsearch
  applies its own ``english`` analyzer to the cleaned text.
"""

from __future__ import annotations

import html
import re
import unicodedata

_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_REPEATED_PUNCT = re.compile(r"([!?.\-_=*#])\1{3,}")
# Keeps technical tokens intact: "tls 1.3", "c++", "cve-2021-44228", "node.js", "s3"
_TOKEN = re.compile(r"[a-z0-9]+(?:[.+#\-][a-z0-9]+)*\+*")

STOPWORDS = frozenset(
    """a an and are as at be been but by can do does for from has have how i if in into is it its
    my of on or our so that the their them then there these this those to was we were what when
    where which while who why will with you your about after before being both did doing each
    few more most other over same some such than too very just also not no nor only own off out
    up down again further once here any all""".split()
)

MIN_CONTENT_CHARS = 20


def clean_text(text: str | None) -> str:
    """Normalise unicode, strip HTML/URLs/control chars and collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _CONTROL.sub(" ", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    return _WHITESPACE.sub(" ", text).strip()


def is_meaningful(text: str, min_chars: int = MIN_CONTENT_CHARS) -> bool:
    """True when the cleaned text has enough alphanumeric content to be worth indexing."""
    return sum(ch.isalnum() for ch in text) >= min_chars


def _light_stem(token: str) -> str:
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str, remove_stopwords: bool = True, stem: bool = True) -> list[str]:
    """Lexical tokenisation for BM25 and overlap features."""
    tokens = _TOKEN.findall(clean_text(text).lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [_light_stem(t) for t in tokens]
    return tokens
