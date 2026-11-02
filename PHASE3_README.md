# NNC Phase 3 experiment harness

No changes to C++ sources. Run `python3 test_metrics.py` for independent reference calculations on four fixed confusion matrices. These tests validate reference formulas, **not** the C++ implementation; the latter still requires direct integration tests.

Run a quick smoke test on the supplied Ionosphere split:

```bash
FOLDS=1 RUNS=1 GENERATIONS=20 FITNESSES='macro_f1' ./run_experiments.sh
```

For ten pre-split folds, provide files named `dataset.1.train`, `dataset.1.test`, ... `dataset.10.train`, `dataset.10.test` in DATA_DIR, then:

```bash
DATA_DIR=examples DATASETS='ionosphere yeast' FOLDS=10 RUNS=30 GENERATIONS=500 ./run_experiments.sh
```

Do not assume that `examples/ionosphere.train` constitutes ten folds. Seeds 1..30 are used per fold and metric. All metrics are parsed from the NNC logs. The harness records wall-clock elapsed seconds at 1-second granularity. It does not record fitness evaluation counts because the current NNC does not expose a counter. Avoid changing experiment settings while reusing the same results directory; the existing log names are not parameter-versioned. For a new run use a new RESULTS_DIR.
