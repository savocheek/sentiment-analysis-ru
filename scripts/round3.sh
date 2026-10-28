#!/usr/bin/env bash
# Round 3 GPU queue (overnight).
# 1. multi-large-big: model_multi continued on ~330k real reviews (train_big.py, resumable checkpoints)
# 2. rubert-tiny2 student distilled from the better teacher (multi / multi-big) — browser-demo candidate
# 3. two more seeds of the current base-student recipe (teacher model_multi, same 100k) — spread
# Strictly one GPU job at a time, 60 s cool-down between trainings. Usage (from the repo root): bash scripts/round3.sh > round3.log 2>&1
# Safe to rerun after a crash: finished steps are skipped, train_big.py resumes from checkpoints/big.
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 OMP_NUM_THREADS=8
PY=${PY:-.venv/Scripts/python.exe}
step() { echo "== $(date +%H:%M) $*"; }
cool() { step "cool-down 60 s"; sleep 60; }

# full scorecard of a final candidate: hard-dev, hard-test (opened once), typos, Kinopoisk, calibration, CPU int8
scorecard() {
  local m=$1
  step "$m scorecard (hard-dev)"; "$PY" evaluate.py "$m" 2>/dev/null | grep '^{' | tee -a experiments.jsonl
  if ! grep -q "\"$m\"" final_test.jsonl; then
    step "$m hard-test (once)"; "$PY" evaluate.py "$m" --hard test 2>/dev/null | grep '^{' | tee -a final_test.jsonl
  fi
  step "$m stress test"; "$PY" stress_test.py "$m" "reports/stress_test_$m.md" | grep -A8 "Устойчивость"
  step "$m kinopoisk"; "$PY" kinopoisk_eval.py "$m" 2>/dev/null | grep "^$m"
  step "$m calibration"; "$PY" calibrate.py "$m" | head -20
  step "$m ONNX (CPU)"
  DEVICE=cpu CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= "$PY" export_onnx.py "$m" "onnx_$m" > "onnx_$m.log" 2>&1 \
    && DEVICE=cpu CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= "$PY" onnx_quality.py "$m" "onnx_$m" >> "onnx_$m.log" 2>&1
  cat "onnx_$m/onnx_report.json"
}

distill() {  # out, args…
  local out=$1; shift
  if [ ! -f "$out/heads.pt" ]; then
    step "distill $out $*"
    "$PY" distill.py --out "$out" "$@" > "distill_$out.log" 2>&1 || { step "!! $out failed"; tail -20 "distill_$out.log"; return 1; }
  fi
  grep -E "^(epoch|test|transfer|student)" "distill_$out.log" | cut -c1-400
}

# 1. multi-large-big
if [ ! -f model_multi_big/multitask.json ]; then
  step "train_big"
  "$PY" train_big.py >> train_big.log 2>&1 || { step "!! train_big failed"; tail -20 train_big.log; }
fi
grep -E "^(big train|train=|start|eval|test|no evaluation)" train_big.log | cut -c1-600
TEACHER=model_multi; CACHE=data/distill_teacher.npz
if [ -f model_multi_big/multitask.json ]; then  # saved only if it beat model_multi on val (mean macro-F1)
  cool
  scorecard model_multi_big
  TEACHER=model_multi_big; CACHE=data/distill_teacher_big.npz
fi
step "teacher for tiny: $TEACHER"

# 2. tiny student
cool
distill model_tiny --teacher "$TEACHER" --cache "$CACHE" --student cointegrated/rubert-tiny2 --epochs 5 --lr 1e-4 --batch 64 \
  && { cool; scorecard model_tiny; }

# 3. seeds of the current base recipe
for s in 1 2; do
  cool
  distill "model_student_s$s" --teacher model_multi --seed "$s" \
    && "$PY" evaluate.py "model_student_s$s" 2>/dev/null | grep '^{' | tee -a experiments.jsonl
done
step done
