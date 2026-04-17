#!/usr/bin/env python3
"""
pipeline/01_build_index.py — Corpus Indexing

Converts PDF corpus to normalized text, chunks it,
builds BM25 index, and (optionally) builds a dense
embedding index for hybrid retrieval.

Usage:
    python pipeline/01_build_index.py \
        --pdf-dir data/corpus/ \
        --outdir  output/step1/

    # With dense index (required for hybrid retrieval):
    python pipeline/01_build_index.py \
        --pdf-dir data/corpus/ \
        --outdir  output/step1/ \
        --dense \
        --dense-model allenai/specter2_base
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


# ── Text normalization ────────────────────────────────────────────────

SYNONYM_MAP = {
    "mtd":                      "mass transport deposit",
    "mtds":                     "mass transport deposits",
    "mass wasting":             "mass transport deposit",
    "mass movement":            "mass transport deposit",
    "submarine landslide":      "mass transport deposit",
    "slope failure":            "slope failure",
}

def normalize_text(text: str) -> str:
    text = text.lower()
    for abbr, full in SYNONYM_MAP.items():
        text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── PDF conversion ────────────────────────────────────────────────────

def pdf_to_text(pdf_path: Path) -> str:
    """
    Convert a single PDF to plain text.
    Uses pdfplumber (preferred) with pymupdf fallback.
    """
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except ImportError:
        pass

    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        return "\n".join(pages)
    except ImportError:
        raise RuntimeError(
            "No PDF library found. Install one:\n"
            "  pip install pdfplumber\n"
            "  pip install pymupdf"
        )


# ── Chunking ──────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> list[dict]:
    """
    Split normalized text into overlapping character-level chunks.
    Tries to break at sentence boundaries ('. ') within the window.
    """
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at sentence boundary within last 100 chars
        if end < len(text):
            boundary = text.rfind('. ', end - 100, end)
            if boundary != -1:
                end = boundary + 1

        chunk_text_str = text[start:end].strip()
        if len(chunk_text_str) > 50:  # skip near-empty chunks
            chunks.append({
                "chunk_id":    f"{doc_id}::{chunk_idx}",
                "doc_id":      doc_id,
                "text":        chunk_text_str,
                "char_start":  start,
                "char_end":    end,
            })
            chunk_idx += 1

        start = end - overlap
        if start >= len(text) - 50:
            break

    return chunks


# ── BM25 index ────────────────────────────────────────────────────────

def build_and_save_bm25(
    chunks: list[dict],
    output_dir: Path,
) -> None:
    """
    Save chunks.jsonl — the BM25 index is rebuilt at query time
    from this file by pipeline/rag/bm25_unified.py.
    (BM25 does not require a serialized index object.)
    """
    out_path = output_dir / "chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"  Saved {len(chunks)} chunks → {out_path}")


# ── Dense index (optional) ────────────────────────────────────────────

def build_and_save_dense_index(
    chunks: list[dict],
    output_dir: Path,
    model_name: str = "allenai/specter2_base",
    batch_size: int = 64,
    device: str = "cuda",
) -> None:
    """
    Encode all chunks into dense vectors using a scientific text encoder.
    Saves:
      output/step1/dense_embeddings.npy   — float32 [N, D] array
      output/step1/dense_model.txt        — model name for reproducibility

    Model choices (in order of preference for geological text):
      allenai/specter2_base     — trained on scientific papers, best semantic match
      BAAI/bge-small-en-v1.5   — faster, smaller, good general-purpose fallback
      allenai/scibert_scivocab_uncased  — already in your env, weakest for retrieval
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  WARNING: sentence-transformers not installed. Skipping dense index.")
        print("  Install with: pip install sentence-transformers")
        return

    print(f"  Loading dense encoder: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    texts = [c["text"] for c in chunks]
    print(f"  Encoding {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,   # cosine similarity = dot product
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    emb_path = output_dir / "dense_embeddings.npy"
    np.save(emb_path, embeddings)
    print(f"  Saved dense embeddings: {embeddings.shape} → {emb_path}")

    # Save model name for reproducibility check in 02_extract_triples.py
    model_path = output_dir / "dense_model.txt"
    model_path.write_text(model_name)
    print(f"  Saved model name → {model_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BM25 + optional dense index from PDF corpus"
    )
    parser.add_argument("--pdf-dir",      required=True,
                        help="Directory containing PDF files")
    parser.add_argument("--outdir",       required=True,
                        help="Output directory (will be created)")
    parser.add_argument("--chunk-size",   type=int, default=800)
    parser.add_argument("--overlap",      type=int, default=200)
    parser.add_argument("--dense",        action="store_true",
                        help="Also build dense embedding index")
    parser.add_argument("--dense-model",  default="allenai/specter2_base",
                        help="HuggingFace model name for dense encoding")
    parser.add_argument("--dense-batch",  type=int, default=64)
    parser.add_argument("--device",       default="cuda",
                        choices=["cuda", "cpu"])
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"ERROR: No PDF files found in {pdf_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(pdf_files)} PDF files")

    # ── Step 1: PDF → normalized text → chunks ────────────────────────
    all_chunks = []
    failed = []

    for pdf_path in pdf_files:
        doc_id = pdf_path.stem
        print(f"  Processing: {doc_id}")
        try:
            raw_text = pdf_to_text(pdf_path)
            norm_text = normalize_text(raw_text)
            chunks = chunk_text(
                norm_text, doc_id,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
            all_chunks.extend(chunks)
            print(f"    → {len(chunks)} chunks")
        except Exception as e:
            print(f"    ERROR: {e}")
            failed.append({"doc_id": doc_id, "error": str(e)})

    print(f"\nTotal chunks: {len(all_chunks)} from {len(pdf_files) - len(failed)} papers")
    if failed:
        print(f"Failed: {len(failed)} papers")
        fail_path = output_dir / "failed_pdfs.json"
        with open(fail_path, "w") as f:
            json.dump(failed, f, indent=2)

    # ── Step 2: Save chunks (BM25 index source) ───────────────────────
    build_and_save_bm25(all_chunks, output_dir)

    # ── Step 3: Dense index (optional) ───────────────────────────────
    if args.dense:
        build_and_save_dense_index(
            all_chunks, output_dir,
            model_name=args.dense_model,
            batch_size=args.dense_batch,
            device=args.device,
        )
    else:
        print("\n  Dense index skipped (pass --dense to build it)")

    # ── Step 4: Write index metadata ─────────────────────────────────
    meta = {
        "n_papers":    len(pdf_files),
        "n_failed":    len(failed),
        "n_chunks":    len(all_chunks),
        "chunk_size":  args.chunk_size,
        "overlap":     args.overlap,
        "dense_built": args.dense,
        "dense_model": args.dense_model if args.dense else None,
    }
    meta_path = output_dir / "index_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Index metadata → {meta_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()