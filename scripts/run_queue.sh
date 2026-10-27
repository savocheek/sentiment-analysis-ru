#!/usr/bin/env bash
# Train + evaluate a queue of experiments; each scorecard is appended to experiments.jsonl.
# Usage (from the repo root): bash scripts/run_queue.sh  (edit the list at the bottom). Logs: train_<out>.log
export PYTHONUNBUFFERED=1 PYTHONUTF8=1PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=${PY:-.venv/Scripts/python.exe}

run() {  # --out must be the last argument
  local out="${@: -1}"
  [ -d "$out" ] && { echo "skip $out (exists)"; return; }
  echo "== $(date +%H:%M) start $out"
  "$PY" train.py "$@" > "train_$out.log" 2>&1 || { echo "!! $out train failed"; tail -5 "train_$out.log"; return; }
  "$PY" evaluate.py "$out" 2>>"train_$out.log" | grep '^{' | tee -a experiments.jsonl
  echo "== $(date +%H:%M) done $out"
}

run --out model_v1
run --augment --out model_v2
run --augment --hard --out model_v3a_gpu
run --augment --hard --base-model seara/rubert-base-cased-russian-sentiment --lr 3e-5 --out model_v3c
run --augment --hard --base-model ai-forever/ruBert-base --out model_v3b
run --augment --hard --base-model ai-forever/ruRoberta-large --lr 1e-5 --batch 8 --out model_v3d
