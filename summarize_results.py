#!/usr/bin/env python3
import csv, sys, math, statistics, collections
src = sys.argv[1] if len(sys.argv)>1 else "results/phase3/results.tsv"
dst = sys.argv[2] if len(sys.argv)>2 else "results/phase3/summary.tsv"
metrics = ["test_accuracy","test_macro_f1","test_balanced_accuracy","test_gmean","test_mcc","seconds"]
groups = collections.defaultdict(list)
with open(src, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if row["status"] != "OK": continue
        try:
            values = [float(row[m]) for m in metrics]
            if not all(math.isfinite(v) for v in values): continue
        except (ValueError, KeyError): continue
        groups[(row["dataset"],row["fitness"])].append(values)
with open(dst,"w",newline="") as f:
    w=csv.writer(f,delimiter="\t")
    w.writerow(["dataset","fitness","n"]+[x+"_"+s for x in metrics for s in ("mean","std")])
    for (dataset,fitness), data in sorted(groups.items()):
        cols=list(zip(*data))
        w.writerow([dataset,fitness,len(data)]+[
            f"{v:.10f}" for col in cols for v in
            (statistics.mean(col), statistics.stdev(col) if len(col)>1 else 0.0)
        ])
print("Summary:",dst)
