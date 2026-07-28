#!/usr/bin/env bash
# run13_pipeline_v2.sh — clean restart version.
# Prerequisites already done: corpus in data/corpus_run13/ (same papers as
# run11 — verify!), protocol frozen & committed (configs/lb_dev.json,
# lb_test.json, run13_protocol_declaration.json).
#
# Usage (from ~/ontogeorag, venv active, inside a salloc with enough RAM):
#   bash analysis_suite/run13_pipeline_v2.sh            # all stages
#   FROM_STAGE=3 bash analysis_suite/run13_pipeline_v2.sh   # resume at 3
#
# Stages: 2=index 3=extract 4=verify 5=clean 6=fusion+metrics 8=analysis
set -e
unset SLURM_PROCID SLURM_NTASKS RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD"
MODEL_LOCAL=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_files_only=True))")
echo "[model resolved] $MODEL_LOCAL"
MODEL_LOCAL=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_files_only=True))")
echo "[model resolved] $MODEL_LOCAL"

RUN=run13
OUT=output/${RUN}
FROM_STAGE=${FROM_STAGE:-2}
N_PAPERS_EXPECTED=${N_PAPERS_EXPECTED:-37}   # set 38 if diagnosis says so
mkdir -p "$OUT"/{step1,pass_a,pass_b,kg}

if [ "$FROM_STAGE" -le 2 ]; then
echo "================ STAGE 2 — build index ================"
n_pdf=$(ls data/corpus_run13/*.pdf | wc -l)
echo "[check] corpus_run13: $n_pdf PDFs (expected $N_PAPERS_EXPECTED)"
[ "$n_pdf" -eq "$N_PAPERS_EXPECTED" ] || { echo "corpus count mismatch — fix corpus first"; exit 1; }
python pipeline/01_build_index.py \
    --pdf-dir data/corpus_run13/ \
    --outdir "$OUT"/step1/
python - << EOF
import json, hashlib
papers, hashes, dup = set(), set(), 0
for line in open("$OUT/step1/chunks.jsonl", encoding="utf-8"):
    r = json.loads(line)
    papers.add(r["doc_id"])
    h = hashlib.md5((r["doc_id"] + r["text"]).encode()).hexdigest()
    dup += h in hashes
    hashes.add(h)
print(f"[check] papers={len(papers)} (expect $N_PAPERS_EXPECTED), "
      f"chunks={len(hashes)}, duplicate-chunks={dup} (expect 0)")
assert dup == 0 and len(papers) == $N_PAPERS_EXPECTED, "INDEX PROBLEM — stop."
EOF
fi

if [ "$FROM_STAGE" -le 3 ]; then
echo "================ STAGE 3 — extraction (dual pass) ================"
for PASS in a b; do
    TEMP=$([ "$PASS" = "a" ] && echo 0.0 || echo 0.3)
    python -u pipeline/02_extract_triples.py \
        --index-dir "$OUT"/step1/ \
        --queries configs/descriptor_queries.jsonl \
        --output "$OUT"/pass_${PASS}/raw_triples.jsonl \
        --schema configs/ontology_schema.json \
        --model "$MODEL_LOCAL" --backend hf \
        --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --bm25-topn 20 --top-k 5 --min-bm25 2.0 \
        --temperature ${TEMP}
done
# If 02's flags differ from run11's exact invocation, edit THIS block to the
# run11 commands verbatim — only --index-dir/--output must point to run13.
fi

if [ "$FROM_STAGE" -le 4 ]; then
echo "================ STAGE 4 — verification ================"
for PASS in a b; do
    python pipeline/03_verify_triples.py \
        --input "$OUT"/pass_${PASS}/raw_triples.jsonl \
        --output "$OUT"/pass_${PASS}/verified_triples.jsonl \
        --model "$MODEL_LOCAL" --backend hf
done
fi

if [ "$FROM_STAGE" -le 5 ]; then
echo "================ STAGE 5 — clean + rule-canon + SciBERT + lexicon ===="
for PASS in a b; do
    python -u pipeline/04_clean_validate.py \
        --input "$OUT"/pass_${PASS}/verified_triples.jsonl \
        --outdir "$OUT"/pass_${PASS}/
    python analysis_suite/05a_rule_canonicalize.py \
        --input "$OUT"/pass_${PASS}/canonical_triples_v5.jsonl
    python analysis_suite/04c_lexicon_enforce.py \
        --input "$OUT"/pass_${PASS}/canonical_triples_v5.rulecanon.jsonl --enforce
done
fi

if [ "$FROM_STAGE" -le 6 ]; then
echo "================ STAGE 6 — fusion + provenance + metrics ============="
python pipeline/06_tiered_fusion.py \
    --iter-a "$OUT"/pass_a/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl \
    --iter-b "$OUT"/pass_b/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl \
    --output "$OUT"/kg/tiered_kg_run13.json
python analysis_suite/06b_attach_provenance.py \
    --kg "$OUT"/kg/tiered_kg_run13.json \
    --pass-a "$OUT"/pass_a/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl \
    --pass-b "$OUT"/pass_b/canonical_triples_v5.rulecanon.lexicon_enforced.jsonl \
    --out "$OUT"/kg/tiered_kg_run13_prov.json
for REF in dev test; do
    python pipeline/07_final_metrics.py \
        --kg "$OUT"/kg/tiered_kg_run13.json \
        --ref configs/lb_${REF}.json \
        --output "$OUT"/kg/metrics_${REF}.json
done
python pipeline/07_final_metrics.py \
    --kg "$OUT"/kg/tiered_kg_run13.json \
    --ref configs/lb_extended_benchmark.json \
    --output "$OUT"/kg/metrics_full34.json
echo ">>> NEXT (manual): M4 battery + 04b on $OUT/kg/tiered_kg_run13_prov.json"
echo ">>> then: FROM_STAGE=8 bash analysis_suite/run13_pipeline_v2.sh"
fi

if [ "$FROM_STAGE" -ge 8 ]; then
echo "================ STAGE 8 — analysis suite ================"
KG_FINAL="$OUT"/kg/tiered_kg_run13_enforced.json
[ -f "$KG_FINAL" ] || KG_FINAL="$OUT"/kg/tiered_kg_run13_prov.json
echo "[analysis on] $KG_FINAL"
cd analysis_suite
bash run_full_analysis.sh ../"$KG_FINAL" ../"$OUT"/step1/chunks.jsonl ../"$OUT"/analysis
cd ..
echo "RUN13 DONE — $OUT/kg/metrics_{dev,test,full34}.json ; $OUT/analysis/"
fi