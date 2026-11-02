#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA_DIR=${DATA_DIR:-"$HOME/Desktop/ERGASIES/FeatureConstruction2/datasets/tenfolding"}
RESULTS_DIR=${RESULTS_DIR:-results/phase3}
FOLDS=${FOLDS:-10}
RUNS=${RUNS:-10}
GENERATIONS=${GENERATIONS:-500}
POPULATION=${POPULATION:-500}
SEED_START=${SEED_START:-1}
FITNESSES=${FITNESSES:-'error balanced_error macro_f1 weighted_f1 gmean macro_recall mcc'}
DATASETS=${DATASETS:-ionosphere}
usage() {
 cat <<'HELP'
Usage: ./run_experiments.sh [options]
  --dataset NAME|all      Dataset name (or all detected datasets)
  --data-dir DIR          Default: ~/Desktop/ERGASIES/FeatureConstruction2/datasets/tenfolding
  --folds N               Default: 10
  --runs N                Default: 10
  --seed N                First seed (default: 1)
  --fitness "LIST"        Space-separated fitness methods
  --generations N         Default: 500
  --chromosomes N         Population (default: 500)
  --population N          Alias for --chromosomes
  --out DIR               Output directory
  --help                  Show this help
Environment variables FOLDS, RUNS, FITNESSES, DATASETS, DATA_DIR etc. still work.
HELP
}
while (($#)); do
 case "$1" in
 --dataset) DATASETS=${2:?}; shift 2;;
 --data-dir) DATA_DIR=${2:?}; shift 2;;
 --folds) FOLDS=${2:?}; shift 2;;
 --runs) RUNS=${2:?}; shift 2;;
 --seed) SEED_START=${2:?}; shift 2;;
 --fitness) FITNESSES=${2:?}; shift 2;;
 --generations) GENERATIONS=${2:?}; shift 2;;
 --chromosomes|--population) POPULATION=${2:?}; shift 2;;
 --out) RESULTS_DIR=${2:?}; shift 2;;
 --help|-h) usage; exit 0;;
 *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
 esac
done
DATA_DIR=${DATA_DIR/#\~/$HOME}
if [[ "$DATASETS" == all ]]; then
 shopt -s nullglob
 declare -A found=()
 for file in "$DATA_DIR"/*.train "$DATA_DIR"/*.1.train "$DATA_DIR"/*_1.train; do
  name=${file##*/}; name=${name%.train}; name=${name%.1}; name=${name%_1}
  found["$name"]=1
 done
 DATASETS="${!found[*]}"
 [[ -n "$DATASETS" ]] || { echo "No datasets in $DATA_DIR" >&2; exit 2; }
fi
# Include the configuration in the result path to prevent unsafe resume collisions.
RESULTS_DIR="$RESULTS_DIR/g${GENERATIONS}_p${POPULATION}_f${FOLDS}"

mkdir -p "$RESULTS_DIR/logs"
TSV="$RESULTS_DIR/results.tsv"
HEADER=$'dataset\tfold\tseed\tfitness\ttrain_fitness\ttrain_accuracy\ttrain_macro_f1\ttrain_balanced_accuracy\ttrain_gmean\ttrain_mcc\ttest_accuracy\ttest_macro_f1\ttest_balanced_accuracy\ttest_gmean\ttest_mcc\tseconds\tstatus\tlog'
if [[ ! -e "$TSV" ]]; then printf '%s\n' "$HEADER" > "$TSV"; fi
if [[ "$(head -n 1 "$TSV")" != "$HEADER" ]]; then
 echo "Unexpected TSV header in $TSV. Use a separate RESULTS_DIR." >&2; exit 2
fi
[[ -x ./NNC ]] || { echo "Missing executable ./NNC" >&2; exit 2; }
extract() {
 awk -F ' = ' -v key="$1" '$1==key {split($2,a," ");v=a[1]} END {if(v!="") print v; else print "NA"}' "$2"
}
# A successful TSV record, not an existing log, is the authoritative resume marker.
completed() {
 awk -F '\t' -v d="$1" -v f="$2" -v s="$3" -v m="$4" \
 'NR>1 && $1==d && $2==f && $3==s && $4==m && $17=="OK" {found=1} END {exit !found}' "$TSV"
}
for dataset in $DATASETS; do
 for ((fold=1;fold<=FOLDS;fold++)); do
  train=""; test=""
  if ((FOLDS==1)); then
    for stem in "$dataset" "$dataset.1" "${dataset}_1"; do
      if [[ -f "$DATA_DIR/$stem.train" && -f "$DATA_DIR/$stem.test" ]]; then
        train="$DATA_DIR/$stem.train"; test="$DATA_DIR/$stem.test"; break
      fi
    done
  else
    for stem in "${dataset}.${fold}" "${dataset}_${fold}"; do
      if [[ -f "$DATA_DIR/$stem.train" && -f "$DATA_DIR/$stem.test" ]]; then
        train="$DATA_DIR/$stem.train"; test="$DATA_DIR/$stem.test"; break
      fi
    done
  fi
  [[ -n "$train" && -n "$test" && -f "$train" && -f "$test" ]] || { echo "Missing: $train or $test" >&2; exit 2; }
  for fitness in $FITNESSES; do
   for ((run=0;run<RUNS;run++)); do
    seed=$((SEED_START+run))
    if completed "$dataset" "$fold" "$seed" "$fitness"; then
      echo "SKIP $dataset fold=$fold fitness=$fitness seed=$seed"; continue
    fi
    log="$RESULTS_DIR/logs/${dataset}_fold${fold}_${fitness}_seed${seed}.log"
    tmp="${log}.tmp.$$"
    echo "RUN $dataset fold=$fold fitness=$fitness seed=$seed"
    start=$(date +%s%N)
    status=OK
    ./NNC --train "$train" --test "$test" --fitness "$fitness" --seed "$seed" \
      --generations "$GENERATIONS" --population "$POPULATION" > "$tmp" 2>&1 || status=FAIL
    end=$(date +%s%N)
    elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN {printf "%.3f", (b-a)/1e9}')
    mv -f "$tmp" "$log"
    for metric in 'TRAIN FITNESS' 'TRAIN ACCURACY' 'TRAIN MACRO F1' \
      'TRAIN BALANCED ACC' 'TRAIN GMEAN' 'TRAIN MCC' 'TEST ACCURACY' \
      'TEST MACRO F1' 'TEST BALANCED ACC' 'TEST GMEAN' 'TEST MCC'; do
      val=$(extract "$metric" "$log")
      if [[ ! "$val" =~ ^-?[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]]; then status=FAIL; fi
    done
    grep -q '^SOLUTION:' "$log" || status=FAIL
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$dataset" "$fold" "$seed" "$fitness" \
      "$(extract 'TRAIN FITNESS' "$log")" "$(extract 'TRAIN ACCURACY' "$log")" \
      "$(extract 'TRAIN MACRO F1' "$log")" "$(extract 'TRAIN BALANCED ACC' "$log")" \
      "$(extract 'TRAIN GMEAN' "$log")" "$(extract 'TRAIN MCC' "$log")" \
      "$(extract 'TEST ACCURACY' "$log")" "$(extract 'TEST MACRO F1' "$log")" \
      "$(extract 'TEST BALANCED ACC' "$log")" "$(extract 'TEST GMEAN' "$log")" \
      "$(extract 'TEST MCC' "$log")" "$elapsed" "$status" "$log" >> "$TSV"
    echo "$status (${elapsed}s)"
   done
  done
 done
done
python3 summarize_results.py "$TSV" "$RESULTS_DIR/summary.tsv"
echo "Results: $TSV"

python3 enrich_metrics.py "$TSV"
