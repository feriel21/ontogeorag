#!/bin/bash
#SBATCH --job-name=run12_hybrid
#SBATCH --partition=convergence
#SBATCH --gres=gpu:a100_3g.40gb:1
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/talbi/ontogeorag/logs/run12_%j.out
#SBATCH --error=/home/talbi/ontogeorag/logs/run12_%j.err

REPO=/home/talbi/ontogeorag
VENV=/home/talbi/kg_test/venv
INDEX=$REPO/output/step1
SCHEMA=$REPO/configs/ontology_schema.json
QUERIES=$REPO/configs/descriptor_queries.jsonl
MODEL=Qwen/Qwen2.5-7B-Instruct
A=$REPO/output/run12_a
B=$REPO/output/run12_b
KG=$REPO/output/run12_kg

mkdir -p $A $B $KG $REPO/logs
cd $REPO
source $VENV/bin/activate
export PYTHONPATH=$REPO:$PYTHONPATH

echo "============================================"
echo " OntoGeoRAG Run 12 — Hybrid Retrieval"
echo " BM25 + BGE-small dense + CrossEncoder"
echo " Node: $(hostname)"
echo " GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo " Date: $(date)"
echo "============================================"

echo "[A1/3] Extract pass A (temp=0.0) — HYBRID..."
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX \
    --schema    $SCHEMA \
    --queries   $QUERIES \
    --output    $A/raw_triples.jsonl \
    --model     $MODEL --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 0.0 \
    --temperature 0.0 \
    --hybrid \
    --hybrid-model BAAI/bge-small-en-v1.5 \
    --fusion-alpha 0.5 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED A1" && exit 1

echo "[A2/3] Verify pass A..."
python -u pipeline/03_verify_triples.py \
    --input  $A/raw_triples.jsonl \
    --output $A/verified_triples.jsonl \
    --model  $MODEL --backend hf
[ $? -ne 0 ] && echo "FAILED A2" && exit 1

echo "[A3/3] Clean pass A..."
python -u pipeline/04_clean_validate.py \
    --input  $A/verified_triples.jsonl \
    --outdir $A
[ $? -ne 0 ] && echo "FAILED A3" && exit 1

echo "[B1/3] Extract pass B (temp=0.3) — HYBRID..."
python -u pipeline/02_extract_triples.py \
    --index-dir $INDEX \
    --schema    $SCHEMA \
    --queries   $QUERIES \
    --output    $B/raw_triples.jsonl \
    --model     $MODEL --backend hf \
    --top-k 5 --bm25-topn 20 --min-bm25 0.0 \
    --temperature 0.3 \
    --hybrid \
    --hybrid-model BAAI/bge-small-en-v1.5 \
    --fusion-alpha 0.5 \
    --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
[ $? -ne 0 ] && echo "FAILED B1" && exit 1

echo "[B2/3] Verify pass B..."
python -u pipeline/03_verify_triples.py \
    --input  $B/raw_triples.jsonl \
    --output $B/verified_triples.jsonl \
    --model  $MODEL --backend hf
[ $? -ne 0 ] && echo "FAILED B2" && exit 1

echo "[B3/3] Clean pass B..."
python -u pipeline/04_clean_validate.py \
    --input  $B/verified_triples.jsonl \
    --outdir $B
[ $? -ne 0 ] && echo "FAILED B3" && exit 1

echo "[5] Fusion..."
python -u pipeline/06_tiered_fusion.py \
    --iter-a $A/canonical_triples_v5.jsonl \
    --iter-b $B/canonical_triples_v5.jsonl \
    --output $KG/tiered_kg_run12.json
[ $? -ne 0 ] && echo "FAILED fusion" && exit 1

echo "[6] Metrics..."
python -u pipeline/07_final_metrics.py \
    --kg     $KG/tiered_kg_run12.json \
    --output $KG/metrics_run12.json

echo "============================================"
echo " Run 12 COMPLETE: $(date)"
echo "============================================"

# Comparaison automatique run11 vs run12
python3 - << 'PYEOF'
import json
from pathlib import Path
from pipeline.rag.constants import normalize_relation
from pipeline.rag.constants import normalize_entity

def load_kg(path):
    kg = json.load(open(path))
    return kg.get("triples", kg) if isinstance(kg, dict) else kg

def compute_recall(triples, ref_path):
    ref = json.load(open(ref_path))
    ref = ref.get("edges", ref) if isinstance(ref, dict) else ref
    hits = 0
    for r in ref:
        rs = normalize_entity(r.get("subject",""))
        rr = normalize_relation(r.get("relation",""))
        ro = normalize_entity(r.get("object",""))
        for t in triples:
            ts = normalize_entity(t.get("subject",""))
            tr = normalize_relation(t.get("relation",""))
            to = normalize_entity(t.get("object",""))
            if rs in ts and ro in to and rr == tr:
                hits += 1
                break
    return hits, len(ref)

REPO = Path("/home/talbi/ontogeorag")
REF  = REPO / "configs/lb_reference_edges.json"

r11 = load_kg(REPO / "output/run11_kg/tiered_kg_run11.json")
r12 = load_kg(REPO / "output/run12_kg/tiered_kg_run12.json")

t1_11 = [t for t in r11 if t.get("tier")==1]
t1_12 = [t for t in r12 if t.get("tier")==1]

h11, total = compute_recall(r11, REF)
h12, _     = compute_recall(r12, REF)

print("\n" + "="*50)
print("  RUN 11 vs RUN 12 COMPARISON")
print("="*50)
print(f"  {'Metric':<30} {'Run11':>8} {'Run12':>8} {'Delta':>8}")
print(f"  {'-'*54}")
print(f"  {'Total triples':<30} {len(r11):>8} {len(r12):>8} {len(r12)-len(r11):>+8}")
print(f"  {'Tier-1 triples':<30} {len(t1_11):>8} {len(t1_12):>8} {len(t1_12)-len(t1_11):>+8}")
print(f"  {'Recall (34-edge)':<30} {h11/total*100:>7.1f}% {h12/total*100:>7.1f}% {(h12-h11)/total*100:>+7.1f}pp")
print(f"  {'Benchmark edges hit':<30} {h11:>8} {h12:>8} {h12-h11:>+8}")
print("="*50)
PYEOF
