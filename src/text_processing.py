"""Nettoyage des textes d'avis."""

import html
import re


_HTML_TAG = re.compile(r"<[^>]+>")
_NON_WORD = re.compile(r"[\W_]+", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def clean_text(text):
    """Normalise un texte pour sa vectorisation : HTML, minuscules et ponctuation."""
    if text is None:
        return ""

    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    text = _NON_WORD.sub(" ", text.lower())
    return _WHITESPACE.sub(" ", text).strip()
