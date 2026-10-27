#!/usr/bin/env bash
# GPU queue after the v3 sweep: multi-task model on the winner, then seed repeats of the two best.
# One GPU job at a time, back to back; scorecards run on CPU in the background (DEVICE=cpu, low priority).
# Usage: bash gpu_queue.sh > gpu_queue.log 2>&1
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=${PY:-.venv/Scripts/python.exe}
declare -A FLAGS=(
  [model_v3a_gpu]="--augment --hard"
  [model_v3b]="--augment --hard --base-model ai-forever/ruBert-base"
  [model_v3c]="--augment --hard --base-model seara/rubert-base-cased-russian-sentiment --lr 3e-5"
  [model_v3d]="--augment --hard --base-model ai-forever/ruRoberta-large --lr 1e-5 --batch 8"
)

busy() {  # python processes still training or scoring
  powershell -NoProfile -c "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | ? CommandLine -match '$1').Count" | tr -d '\r'
}
cpu_eval() {  # background scorecard on CPU → experiments.jsonl
  (DEVICE=cpu OMP_NUM_THREADS=8 "$PY" evaluate.py "$1" 2>>"eval_$1.log" | grep '^{' >> experiments.jsonl
   echo "== $(date +%H:%M) scored $1") &
}
ranked() {  # v3 models by hard_dev_bal with real_f1 >= 0.779 (NEXT_STEPS rule), best first
  "$PY" -c "
import json
rows = [json.loads(l) for l in open('experiments.jsonl', encoding='utf-8') if l.strip()]
best = {r['model']: r for r in rows if r['model'] in ('model_v3a_gpu', 'model_v3b', 'model_v3c', 'model_v3d') and r['real_f1'] >= 0.779}
print(' '.join(sorted(best, key=lambda m: -best[m]['hard_dev_bal'])))"
}

echo "== $(date +%H:%M) waiting for other GPU jobs"
while [ "$(busy 'train(_multi)?\.py|evaluate\.py')" != "0" ]; do sleep 20; done
if [ ! -f model_v3d/config.json ]; then  # interrupted by a reboot; scored on GPU right away, the ranking needs it
  echo "== $(date +%H:%M) start model_v3d"
  "$PY" train.py ${FLAGS[model_v3d]} --out model_v3d > train_model_v3d.log 2>&1     && "$PY" evaluate.py model_v3d 2>>train_model_v3d.log | grep '^{' | tee -a experiments.jsonl || echo "!! model_v3d failed"
fi
read -r -a TOP <<< "$(ranked)"
echo "== $(date +%H:%M) ranking: ${TOP[*]}"
W=${TOP[0]}

echo "== $(date +%H:%M) start model_multi on $W"
"$PY" train_multi.py --base-model "$W" --augment --hard --out model_multi > multi_model_multi.log 2>&1 \
  && cpu_eval model_multi || echo "!! model_multi failed"
echo "== $(date +%H:%M) done model_multi"

for m in "${TOP[@]:0:2}"; do
  for s in 1 2; do
    out="${m}_s$s"
    [ -d "$out" ] && continue
    echo "== $(date +%H:%M) start $out"
    "$PY" train.py ${FLAGS[$m]} --seed "$s" --out "$out" > "train_$out.log" 2>&1 && cpu_eval "$out" || echo "!! $out failed"
  done
done
wait
echo "== $(date +%H:%M) queue finished"
