# NNC Phase 3.1
Run `FOLDS=1 RUNS=1 GENERATIONS=20 FITNESSES=macro_f1 ./run_experiments.sh`.
For 10 folds, files must be `examples/<dataset>.<fold>.train` and `.test`.
Set `DATASETS="ionosphere yeast"` for multiple datasets.
Resume skips only keys with a successful TSV row. Failed runs are retried and appended.
Do not run two script instances writing to the same RESULTS_DIR simultaneously.
The summary is computed from all successful TSV rows, including any previous successful
duplicates. Avoid changing experiment parameters in the same RESULTS_DIR: the key
does not include generations or population. Use separate RESULTS_DIR values.
`seconds` is wall-clock time with millisecond display resolution.
