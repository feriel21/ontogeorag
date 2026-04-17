#!/bin/bash
#SBATCH --job-name=build_dense_idx
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/build_dense_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/build_dense_%j.err

VENV=/home/talbi/kg_test/venv
INDEX=$REPO/output/step1

mkdir -p /home/talbi/ontogeorag/logs
source $VENV/bin/activate

echo "Node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Date: $(date)"
echo "Chunks: $(wc -l < $INDEX/chunks.jsonl)"

python - << PYEOF
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX = Path("$REPO/output/step1")

# Load existing chunks
chunks = [json.loads(l) for l in open(INDEX / "chunks.jsonl") if l.strip()]
print(f"Loaded {len(chunks)} chunks")

# Encode
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
embeddings = model.encode(
    [c["text"] for c in chunks],
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
).astype("float32")

# Save
np.save(INDEX / "dense_embeddings.npy", embeddings)
(INDEX / "dense_model.txt").write_text("BAAI/bge-small-en-v1.5")
print(f"Saved {embeddings.shape} -> {INDEX}/dense_embeddings.npy")
PYEOF

echo "Done: $(date)"
ls -lh $INDEX/dense_embeddings.npy $INDEX/dense_model.txt
