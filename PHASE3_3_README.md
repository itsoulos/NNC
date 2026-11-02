# NNC Phase 3.3 — Precision / Recall

Default parameters: 500 generations, 500 chromosomes, 10 runs, 10 folds.

New output metrics printed by NNC: TRAIN/TEST MACRO PRECISION, WEIGHTED PRECISION, MACRO RECALL, WEIGHTED RECALL. Macro averages follow existing NNC convention (classes present in ground truth). The confusion matrix is unchanged.

`run_experiments.sh` now calls `enrich_metrics.py` after experiments, preserving existing `results.tsv` and `summary.tsv` for backward compatibility. It generates:

- `results_extended.tsv`: original fields plus train/test macro/weighted precision and recall, weighted F1.
- `summary_extended.tsv`: mean and std for test precision, recall, F1 and existing metrics.
- `per_class_metrics.tsv`: precision/recall/F1/support for each class in each run and split.

To enrich existing runs without re-running NNC:

```bash
python3 enrich_metrics.py results/phase3/g100_p500_f10/results.tsv
```

If experiments are currently running, do not modify their `results.tsv` while the process is writing; run enrichment when the process finishes. Original files are never overwritten by enrichment.

Recompile NNC after unpacking to get the new metrics printed in logs. The enrichment utility works on **old** logs too, without recompiling.
