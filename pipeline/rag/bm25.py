# rag/bm25.py
# -*- coding: utf-8 -*-

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple


def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split(" ")


@dataclass
class BM25Index:
    chunks: List[dict]                   # {"chunk_id","doc_id","text","meta"}
    df: Dict[str, int]                   # document frequency
    doc_len: Dict[str, int]              # length per chunk
    tf: Dict[str, Counter]               # term frequency per chunk_id
    avgdl: float
    k1: float = 1.5
    b: float = 0.75


def build_bm25_index(chunks: List[dict]) -> BM25Index:
    df = defaultdict(int)
    tf = {}
    doc_len = {}

    for ch in chunks:
        cid = ch["chunk_id"]
        toks = tokenize(ch["text"])
        doc_len[cid] = len(toks)
        c = Counter(toks)
        tf[cid] = c
        for term in c.keys():
            df[term] += 1

    avgdl = sum(doc_len.values()) / max(1, len(doc_len))
    return BM25Index(chunks=chunks, df=dict(df), doc_len=doc_len, tf=tf, avgdl=avgdl)


def bm25_score(index: BM25Index, query: str, chunk_id: str) -> float:
    toks = tokenize(query)
    N = len(index.doc_len)
    score = 0.0
    dl = index.doc_len[chunk_id]
    for term in toks:
        if term not in index.df:
            continue
        df = index.df[term]
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        f = index.tf[chunk_id].get(term, 0)
        denom = f + index.k1 * (1 - index.b + index.b * dl / index.avgdl)
        score += idf * (f * (index.k1 + 1) / max(1e-9, denom))
    return score


def retrieve(index: BM25Index, query: str, top_k: int = 6) -> List[Tuple[float, dict]]:
    scored = []
    for ch in index.chunks:
        cid = ch["chunk_id"]
        s = bm25_score(index, query, cid)
        if s > 0:
            scored.append((s, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]
