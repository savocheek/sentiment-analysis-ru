#!/usr/bin/env bash
# Round 2 GPU queue: calibration and stress test of the final model, distillation into ruBert-base, student scorecards.
# Strictly one GPU job at a time, 60 s cool-down between trainings. Usage: bash round2.sh > round2.log 2>&1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 OMP_NUM_THREADS=8
PY=${PY:-.venv/Scripts/python.exe}
busy() { powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? CommandLine -match '$1').Count" | tr -d '\r'; }
step() { echo "== $(date +%H:%M) $*"; }

step "waiting for kinopoisk_eval"
while [ "$(busy 'kinopoisk_eval|train|distill')" != "0" ]; do sleep 20; done

[ -f reports/calibration.json ] || { step calibrate model; "$PY" calibrate.py model; }
step "stress test model"; "$PY" stress_test.py model reports/stress_test.md | grep -A8 "Устойчивость"
step "cool-down 60 s"; sleep 60

if [ ! -f model_student/heads.pt ]; then
  step "distill"; "$PY" distill.py > distill.log 2>&1 || { step "!! distill failed"; tail -20 distill.log; exit 1; }
  grep -E "^(epoch|test|transfer)" distill.log
fi
step "cool-down 60 s"; sleep 60

step "student scorecard (hard-dev)"; "$PY" evaluate.py model_student 2>/dev/null | grep '^{' | tee -a experiments.jsonl
step "student stress test"; "$PY" stress_test.py model_student reports/stress_test_student.md | grep -A8 "Устойчивость"
step "student kinopoisk"; "$PY" kinopoisk_eval.py model_student 2>/dev/null | grep "^model_student"
step "student calibration"; "$PY" calibrate.py model_student | head -20
step "student ONNX (CPU)"; "$PY" export_onnx.py model_student onnx_student > onnx_student.log 2>&1; cat onnx_student/onnx_report.json
step done
