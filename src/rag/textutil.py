"""中文分词与 BM25。零模型依赖时，这比哈希向量更适合专有名词检索。"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import jieba
from langchain_core.documents import Document

_STOP = {
    "的", "是", "吗", "哪", "呢", "了", "和", "与", "在", "有", "为", "请", "一下",
    "什么", "多少", "怎么", "如何", "能", "用来", "可以", "是否", "？", "?",
}


def load_user_dict(path: Path | None = None) -> None:
    terms = path or Path(__file__).resolve().parents[2] / "data" / "terms.txt"
    if not terms.exists():
        return
    for line in terms.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#"):
            jieba.add_word(word)


load_user_dict()


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in jieba.cut_for_search(text.lower()):
        tok = tok.strip()
        if not tok or tok in _STOP or tok.isspace():
            continue
        if re.fullmatch(r"[\W_]+", tok):
            continue
        tokens.append(tok)
    return tokens


def ngrams(text: str, sizes: tuple[int, ...] = (2, 3, 4)) -> set[str]:
    compact = "".join(text.lower().split())
    latin = set(re.findall(r"[a-z0-9]+", compact))
    grams: set[str] = set(latin)
    for n in sizes:
        if len(compact) < n:
            continue
        grams.update(compact[i : i + n] for i in range(len(compact) - n + 1))
    return grams


def lexical_overlap(query: str, passage: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    return len(q & set(tokenize(passage))) / len(q)


class BM25Index:
    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._tokenized = [tokenize(d.page_content) for d in docs]
        self.avgdl = sum(len(t) for t in self._tokenized) / max(len(self._tokenized), 1)
        df: Counter[str] = Counter()
        for tokens in self._tokenized:
            df.update(set(tokens))
        n = len(docs)
        self.idf = {term: math.log((n - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()}

    def search(self, query: str, k: int) -> list[Document]:
        q_terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for i, tokens in enumerate(self._tokenized):
            tf = Counter(tokens)
            dl = len(tokens) or 1
            score = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += idf * freq * (self.k1 + 1) / denom
            header = tokenize(self.docs[i].page_content.split("\n", 1)[0])
            score += sum(self.idf.get(term, 0.0) * 2.0 for term in q_terms if term in header)
            scored.append((score, i))
        scored.sort(reverse=True)
        return [self.docs[i] for score, i in scored[:k] if score > 0]
