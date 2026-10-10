"""Multilingual Medical Tokenizer for ViBioMIR.

Handles Vietnamese diacritics, Chinese (CJK) segmentation via jieba,
medical terms, drug names, ICD codes, dosage units, and numbers.
"""

import re
import unicodedata

import jieba

# Pre-compile CJK detection regex
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

# Punctuation to clean out (excluding internal hyphens and periods)
PUNCT_CLEAN_PATTERN = re.compile(r"[/,;:()\[\]{}!?\"\'<>*&^%$#@~`+=|\\_]+")

# Token extraction pattern: alphanumeric words that may include internal hyphens or periods
TOKEN_PATTERN = re.compile(r"\b\w+(?:[-.]\w+)*\b")


class MultilingualMedicalTokenizer:
    """Tokenizer designed for multilingual biomedical and clinical retrieval."""

    def __init__(self, expand_compounds: bool = True) -> None:
        self.expand_compounds = expand_compounds

    def tokenize(self, text: str | None) -> list[str]:
        """Tokenize arbitrary text into normalized tokens."""
        if not text:
            return []

        # 1. Unicode normalization (NFKC ensures uniform accents & full-width chars)
        norm_text = unicodedata.normalize("NFKC", text)

        # 2. Chinese segmentation: segment with jieba if CJK characters are present
        if CJK_PATTERN.search(norm_text):
            norm_text = " ".join(jieba.cut(norm_text))

        # 3. Clean syntax characters (slashes, brackets, math symbols)
        norm_text = PUNCT_CLEAN_PATTERN.sub(" ", norm_text)

        # 4. Extract word and code tokens
        matches = TOKEN_PATTERN.findall(norm_text)

        tokens: list[str] = []
        for raw in matches:
            tok = raw.lower().strip(".-")
            if not tok:
                continue
            tokens.append(tok)

            # 5. Compound expansion: for tokens like 'beta-hcg', 'covid-19', 'c00.0',
            # also emit constituent sub-words so queries matching partial parts succeed.
            if self.expand_compounds and ("-" in tok or "." in tok):
                subparts = re.split(r"[-.]", tok)
                for part in subparts:
                    if part and part != tok:
                        tokens.append(part)

        return tokens

    def tokenize_to_str(self, text: str | None) -> str:
        """Tokenize text and return space-separated token string for indexing."""
        return " ".join(self.tokenize(text))

    def prepare_query_terms(self, query: str | None) -> list[str]:
        """Prepare clean token list for Tantivy query execution."""
        tokens = self.tokenize(query)
        # Tantivy query syntax: escape hyphens to prevent negation interpretation
        safe_terms = [t.replace("-", r"\-") for t in tokens]
        return safe_terms

    def build_query_string(self, query: str | None) -> str:
        """Construct a safe space-separated query string for Tantivy."""
        terms = self.prepare_query_terms(query)
        return " ".join(terms)
