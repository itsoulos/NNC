#!/usr/bin/env bash
set -eu

NNC=${NNC:-./NNC}
TRAIN=${TRAIN:-examples/ionosphere.train}
TEST=${TEST:-examples/ionosphere.test}
SEED=${SEED:-1}
GENERATIONS=${GENERATIONS:-100}
OUT=${OUT:-fitness_checks}
mkdir -p "$OUT"

if [ ! -x "$NNC" ]; then echo "Executable not found: $NNC" >&2; exit 1; fi
for fitness in error balanced_error macro_f1 weighted_f1 gmean macro_recall mcc; do
  echo "Running $fitness ..."
  "$NNC" -p "$TRAIN" -t "$TEST" --fitness "$fitness" --seed "$SEED" -n "$GENERATIONS" > "$OUT/$fitness.log" 2>&1
  python3 - "$OUT/$fitness.log" "$fitness" <<'PY'
import sys,re,math
s=open(sys.argv[1]).read(); name=sys.argv[2]
def get(key):
    m=re.search(r'^'+re.escape(key)+r'\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)',s,re.M)
    if not m: raise ValueError('Missing '+key)
    return float(m.group(1))
f=get('TRAIN FITNESS')
metric={'error':lambda:-get('TRAIN CLASS ERROR')/100,
        'balanced_error':lambda:get('TRAIN BALANCED ACC')-1,
        'macro_f1':lambda:get('TRAIN MACRO F1'),
        'weighted_f1':lambda:get('TRAIN WEIGHTED F1'),
        'gmean':lambda:get('TRAIN GMEAN'),
        'macro_recall':lambda:get('TRAIN BALANCED ACC'),
        'mcc':lambda:get('TRAIN MCC')}[name]()
# Metric reporting uses 10 decimals, while class error is printed with 6.
tol=1e-6
ok=math.isfinite(f) and math.isfinite(metric) and abs(f-metric)<tol
print(f'{name:16} fitness={f: .10f} expected={metric: .10f} diff={abs(f-metric):.3g} {"PASS" if ok else "FAIL"}')
if not ok: sys.exit(2)
PY
done
echo "All seven fitness checks passed. Logs: $OUT/"
