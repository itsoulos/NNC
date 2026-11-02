# NNC Phase 3.2 — QGenClass-style experiment CLI

Default dataset folder: `~/Desktop/ERGASIES/FeatureConstruction2/datasets/tenfolding`.

Examples:

```bash
./run_experiments.sh --dataset ionosphere --folds 10 --runs 30 --generations 500 --chromosomes 500 --fitness 'macro_f1 gmean mcc'
./run_experiments.sh --dataset all --folds 10 --runs 30
./run_experiments.sh --dataset ionosphere --folds 1 --runs 3 --out results/smoke
```

Accepted fold filenames: `ionosphere.1.train`/`.test` or `ionosphere_1.train`/`.test`.
With `--folds 1`, also accepts `ionosphere.train`/`.test`.

`--fitness` is a space-separated list; `--seed` is the first seed.
Old environment variable controls remain supported. Output is separated by generation,
population and fold settings. For other changed settings use a different `--out`.
Do not run simultaneous writers to the same results directory.
