#!/usr/bin/env bash
# Round 3, resumed after train_big (model_multi_big already trained and saved).
# multi-big is worse than model_multi on dev sentiment (val F1 0.787 vs 0.819) and stars, so:
# - its scorecard runs for the report only, without hard-test (hard-test stays unopened for it);
# - the tiny student is distilled from model_multi.
# Then seeds of the base recipe, then the WB stage-2 queue. Usage: bash scripts/round3_resume.sh >> round3.log 2>&1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 OMP_NUM_THREADS=8
PY=${PY:-.venv/Scripts/python.exe}
step() { echo "== $(date +%H:%M) $*"; }
cool() { step "cool-down 60 s"; sleep 60; }
CPU="env DEVICE=cpu CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES="

scorecard() {  # model, [--no-hard-test]
  local m=$1
  step "$m scorecard (hard-dev)"; "$PY" evaluate.py "$m" 2>/dev/null | grep '^{' | tee -a experiments.jsonl
  if [ "$2" != "--no-hard-test" ] && ! grep -q "\"$m\"" final_test.jsonl; then
    step "$m hard-test (once)"; "$PY" evaluate.py "$m" --hard test 2>/dev/null | grep '^{' | tee -a final_test.jsonl
  fi
  step "$m stress test"; "$PY" stress_test.py "$m" "reports/stress_test_$m.md" | grep -A8 "Устойчивость"
  step "$m kinopoisk"; "$PY" kinopoisk_eval.py "$m" 2>/dev/null | grep "^$m"
  step "$m calibration"; "$PY" calibrate.py "$m" | head -20
  step "$m ONNX (CPU)"
  $CPU "$PY" export_onnx.py "$m" "onnx_$m" > "onnx_$m.log" 2>&1 && $CPU "$PY" onnx_quality.py "$m" "onnx_$m" >> "onnx_$m.log" 2>&1
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

step "resumed: multi-big scorecard for the report (no hard-test)"
scorecard model_multi_big --no-hard-test
step "teacher for tiny: model_multi"

cool
distill model_tiny --teacher model_multi --cache data/distill_teacher.npz --student cointegrated/rubert-tiny2 --epochs 5 --lr 1e-4 --batch 64 \
  && { cool; scorecard model_tiny; }

for s in 1 2; do
  cool
  distill "model_student_s$s" --teacher model_multi --seed "$s" \
    && "$PY" evaluate.py "model_student_s$s" 2>/dev/null | grep '^{' | tee -a experiments.jsonl
done
step done

cd ../review-insights-wb && bash scripts/stage2.sh > stage2.log 2>&1
