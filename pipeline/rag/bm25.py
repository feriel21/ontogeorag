# pipeline/rag/hybrid_retriever.py

from dataclasses import dataclass
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    bm25_score: float
    dense_score: float
    fused_score: float
    rerank_score: float | None = None


def build_dense_index(
    chunks: list[dict],
    model_name: str = "allenai/specter2_base",  # better than SciBERT for retrieval
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[np.ndarray, SentenceTransformer]:
    """
    Encode all chunks into dense vectors.
    Returns: (embeddings [N, D], model)
    Use specter2_base: trained on scientific paper retrieval,
    handles geological vocabulary better than general-purpose models.
    Alternative: 'BAAI/bge-small-en-v1.5' if memory-constrained.
    """
    model = SentenceTransformer(model_name, device=device)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32), model


def hybrid_retrieve(
    query: str,
    chunks: list[dict],
    bm25_index,                    # existing BM25Index from rag/bm25.py
    dense_embeddings: np.ndarray,  # precomputed from build_dense_index()
    dense_model: SentenceTransformer,
    reranker: CrossEncoder,
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
    fusion_alpha: float = 0.5,     # weight: 0=BM25 only, 1=dense only
    final_top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    1. BM25 top-50 + dense top-50 → union candidate set
    2. Reciprocal Rank Fusion of BM25 and dense scores
    3. Cross-encoder re-rank → top-5
    """
    # --- BM25 candidates ---
    bm25_results = bm25_retrieve(bm25_index, query, top_k=bm25_top_k)
    bm25_map = {r["chunk_id"]: r["score"] for r in bm25_results}

    # --- Dense candidates ---
    q_emb = dense_model.encode(
        [query], normalize_embeddings=True
    ).astype(np.float32)
    cos_scores = (dense_embeddings @ q_emb.T).squeeze()
    dense_top_idx = np.argsort(cos_scores)[-dense_top_k:][::-1]
    dense_map = {
        chunks[i]["chunk_id"]: float(cos_scores[i])
        for i in dense_top_idx
    }

    # --- Reciprocal Rank Fusion ---
    candidate_ids = set(bm25_map) | set(dense_map)
    fused = {}
    bm25_ranked = sorted(bm25_map, key=bm25_map.get, reverse=True)
    dense_ranked = sorted(dense_map, key=dense_map.get, reverse=True)

    for cid in candidate_ids:
        bm25_rank = bm25_ranked.index(cid) + 1 if cid in bm25_map else bm25_top_k + 1
        dense_rank = dense_ranked.index(cid) + 1 if cid in dense_map else dense_top_k + 1
        # RRF formula: 1/(k+rank), k=60 standard
        fused[cid] = (1 - fusion_alpha) * (1 / (60 + bm25_rank)) \
                   +       fusion_alpha  * (1 / (60 + dense_rank))

    # Take top-100 by fused score for re-ranking
    top_fused = sorted(fused, key=fused.get, reverse=True)[:100]
    chunk_map = {c["chunk_id"]: c for c in chunks}

    # --- Cross-encoder re-rank (batched) ---
    pairs = [(query, chunk_map[cid]["text"]) for cid in top_fused if cid in chunk_map]
    rerank_scores = reranker.predict(pairs, batch_size=32)

    results = []
    for i, cid in enumerate([c for c in top_fused if c in chunk_map]):
        c = chunk_map[cid]
        results.append(RetrievedChunk(
            chunk_id=cid,
            doc_id=c.get("doc_id", ""),
            text=c["text"],
            bm25_score=bm25_map.get(cid, 0.0),
            dense_score=dense_map.get(cid, 0.0),
            fused_score=fused[cid],
            rerank_score=float(rerank_scores[i]),
        ))

    results.sort(key=lambda x: x.rerank_score, reverse=True)
    return results[:final_top_k]