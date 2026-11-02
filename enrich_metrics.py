#!/usr/bin/env python3
"""Recover per-class and aggregate precision/recall from saved NNC confusion matrices.
Usage: python3 enrich_metrics.py results/phase3/g500_p500_f10/results.tsv
Writes results_extended.tsv, summary_extended.tsv and per_class_metrics.tsv alongside source.
Does not modify original TSV or logs.
"""
import csv
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def cm_from_log(path, prefix):
    lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    marker = f'{prefix} CONFUSION MATRIX'
    for i, line in enumerate(lines):
        if not line.startswith(marker):
            continue
        labels_line = lines[i + 1] if i + 1 < len(lines) else ''
        if not labels_line.startswith(f'{prefix} LABELS:'):
            continue
        labels = labels_line.split(':', 1)[1].split()
        matrix = []
        for line in lines[i + 2:i + 2 + len(labels)]:
            match = re.fullmatch(rf'{prefix} CM\s+(\S+):\s+(.+)', line.strip())
            if not match:
                break
            try:
                values = [int(v) for v in match.group(2).split()]
            except ValueError:
                break
            if len(values) != len(labels):
                break
            matrix.append(values)
        if len(matrix) == len(labels):
            return labels, matrix
    raise ValueError(f'{prefix} confusion matrix not found')


def calculate(labels, cm):
    n = len(labels)
    support = [sum(row) for row in cm]
    pred = [sum(cm[i][j] for i in range(n)) for j in range(n)]
    total = sum(support)
    if total == 0:
        raise ValueError('empty confusion matrix')
    classes = []
    for j in range(n):
        tp = cm[j][j]
        precision = tp / pred[j] if pred[j] else 0.0
        recall = tp / support[j] if support[j] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes.append(dict(label=labels[j], support=support[j], precision=precision, recall=recall, f1=f1))
    # Match the NNC convention: macro averages include classes present in the actual set.
    present = [c for c in classes if c['support'] > 0]
    metrics = {}
    for key in ('precision', 'recall', 'f1'):
        metrics[f'macro_{key}'] = sum(c[key] for c in present) / len(present)
        metrics[f'weighted_{key}'] = sum(c[key] * c['support'] for c in classes) / total
    return metrics, classes


def main(src):
    with src.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        original_columns = reader.fieldnames
        if not original_columns:
            raise ValueError('Empty TSV')
        rows = list(reader)
    extra = [f'{side}_{metric}' for side in ('train', 'test') for metric in
             ('macro_precision', 'weighted_precision', 'macro_recall', 'weighted_recall', 'weighted_f1')]
    columns = list(original_columns) + [x for x in extra if x not in original_columns]
    out_rows, per_class, failures = [], [], 0
    for row in rows:
        new = dict(row)
        if row.get('status') == 'OK':
            log = Path(row['log'])
            if not log.is_file():
                candidate = src.parent / row['log']
                if candidate.is_file():
                    log = candidate
            try:
                for side in ('train', 'test'):
                    labels, matrix = cm_from_log(log, side.upper())
                    metrics, classes = calculate(labels, matrix)
                    for metric in ('macro_precision', 'weighted_precision', 'macro_recall', 'weighted_recall', 'weighted_f1'):
                        new[f'{side}_{metric}'] = f'{metrics[metric]:.10f}'
                    for c in classes:
                        per_class.append({k: row.get(k, '') for k in ('dataset', 'fold', 'seed', 'fitness')} |
                                         {'split': side, **{k: f'{v:.10f}' if isinstance(v, float) else v for k, v in c.items()}})
            except (OSError, ValueError, IndexError) as e:
                failures += 1
                print(f'WARNING: {log}: {e}', file=sys.stderr)
        out_rows.append(new)
    out = src.with_name('results_extended.tsv')
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter='\t', extrasaction='ignore')
        w.writeheader(); w.writerows(out_rows)
    per_path = src.with_name('per_class_metrics.tsv')
    with per_path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset','fold','seed','fitness','split','label','support','precision','recall','f1'], delimiter='\t')
        w.writeheader(); w.writerows(per_class)
    summary_metrics = ['test_accuracy', 'test_macro_precision', 'test_weighted_precision',
                       'test_macro_recall', 'test_weighted_recall', 'test_macro_f1',
                       'test_weighted_f1', 'test_balanced_accuracy', 'test_gmean', 'test_mcc', 'seconds']
    groups = defaultdict(list)
    for row in out_rows:
        if row.get('status') != 'OK':
            continue
        try:
            values = [float(row[m]) for m in summary_metrics]
            if not all(math.isfinite(v) for v in values):
                continue
        except (ValueError, KeyError, TypeError):
            continue
        groups[(row['dataset'], row['fitness'])].append(values)
    summ = src.with_name('summary_extended.tsv')
    with summ.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['dataset', 'fitness', 'n'] + [f'{m}_{s}' for m in summary_metrics for s in ('mean','std')])
        for (dataset, fitness), vals in sorted(groups.items()):
            cols = list(zip(*vals))
            w.writerow([dataset, fitness, len(vals)] +
                       [f'{v:.10f}' for col in cols for v in (statistics.mean(col), statistics.stdev(col) if len(col) > 1 else 0.0)])
    print(f'Extended results: {out}\nExtended summary: {summ}\nPer-class metrics: {per_path}')
    print(f'Rows: {len(rows)}; complete: {sum(len(v) for v in groups.values())}; missing/incomplete logs: {failures}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit('Usage: python3 enrich_metrics.py path/to/results.tsv')
    main(Path(sys.argv[1]))
