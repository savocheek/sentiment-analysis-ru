#!/usr/bin/env bash
# After all training: score the models that still lack a scorecard (GPU is free now, one job at a time),
# then print the seed summary used to pick the winner.  Usage: bash finalize.sh
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 TQDM_DISABLE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 OMP_NUM_THREADS=8
PY=${PY:-.venv/Scripts/python.exe}

for m in model_multi model_v3d_s1 model_v3d_s2 model_v3b_s1 model_v3b_s2; do
  [ -f "$m/config.json" ] || continue
  grep -q "\"model\": \"$m\"" experiments.jsonl && continue
  echo "== $(date +%H:%M) scoring $m"
  "$PY" evaluate.py "$m" 2>>"eval_$m.log" | grep '^{' | tee -a experiments.jsonl
done

"$PY" - <<'EOF'
import json, statistics as st
rows = {}
for l in open("experiments.jsonl", encoding="utf-8"):
    if l.strip():
        r = json.loads(l)
        rows[r["model"]] = r  # last scorecard per model
fams = {"v3d ruRoberta-large": ["model_v3d", "model_v3d_s1", "model_v3d_s2"],
        "v3b ruBert-base": ["model_v3b", "model_v3b_s1", "model_v3b_s2"],
        "multi (v3d + 1–5★ + категория)": ["model_multi"]}
print("| Семейство | seed | real_f1 | hard_dev_bal |")
print("|---|---|---|---|")
for name, ms in fams.items():
    got = [rows[m] for m in ms if m in rows]
    if not got:
        continue
    f1, hb = [r["real_f1"] for r in got], [r["hard_dev_bal"] for r in got]
    sd = lambda x: f" ± {st.stdev(x):.3f}" if len(x) > 1 else ""
    print(f"| {name} | {len(got)} | {st.mean(f1):.4f}{sd(f1)} | {st.mean(hb):.3f}{sd(hb)} |")
EOF
