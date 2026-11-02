#!/usr/bin/env python3
"""NNC report: read results.tsv or results_extended.tsv, optionally enrich from logs.
Usage: python3 report_logs.py results/phase3/g500_p500_f10/results.tsv
       python3 report_logs.py .../results_extended.tsv --runs 10 --folds 10
Output: report_runs.tsv, report_folds.tsv, report_summary.tsv, report_completeness.tsv.
"""
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ['accuracy','macro_precision','weighted_precision','macro_recall',
           'weighted_recall','macro_f1','weighted_f1','balanced_accuracy','gmean','mcc']
SPLITS = ('train','test')

def read_tsv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f,delimiter='\t')
        if not reader.fieldnames: raise ValueError('Empty input')
        return list(reader),reader.fieldnames

def write_tsv(path,columns,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,delimiter='\t',extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)

def finite(row,key):
    try:
        x=float(row.get(key,''))
        return x if math.isfinite(x) else None
    except (ValueError,TypeError): return None

def stats(values):
    if not values:return ('','')
    return (f'{statistics.mean(values):.10f}',
            f'{statistics.stdev(values):.10f}' if len(values)>1 else '0.0000000000')

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input',type=Path,help='results.tsv or results_extended.tsv')
    ap.add_argument('--folds',type=int,default=10)
    ap.add_argument('--runs',type=int,default=10)
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    if args.folds<1 or args.runs<1:ap.error('--folds and --runs must be positive')
    rows,columns=read_tsv(args.input)
    needed={'dataset','fitness','fold','seed','status'}
    if not needed.issubset(columns):ap.error('Missing columns: '+', '.join(sorted(needed-set(columns))))
    out=args.out or args.input.parent
    out.mkdir(parents=True,exist_ok=True)
    # Newest successful row for a dataset/fold/seed/fitness wins; avoid duplicate reruns.
    unique={}
    failures=defaultdict(int)
    for row in rows:
        key=tuple(row.get(x,'') for x in ('dataset','fitness','fold','seed'))
        if row.get('status')=='OK':unique[key]=row
        else:failures[key[:2]]+=1
    ok=list(unique.values())
    all_keys=sorted({(r['dataset'],r['fitness']) for r in rows})
    metric_keys=[f'{split}_{m}' for split in SPLITS for m in METRICS]
    if 'seconds' in columns:metric_keys.append('seconds')
    # Only metrics that appear in the input are included. Missing values stay missing.
    metric_keys=[m for m in metric_keys if m in columns]
    run_columns=list(dict.fromkeys(columns))
    write_tsv(out/'report_runs.tsv',run_columns,sorted(ok,key=lambda r:(r['dataset'],r['fitness'],int(r['fold']),int(r['seed']))))
    by_fold=defaultdict(list)
    for row in ok:by_fold[(row['dataset'],row['fitness'],row['fold'])].append(row)
    fold_cols=['dataset','fitness','fold','n_runs']+[f'{m}_{s}' for m in metric_keys for s in ('mean','std')]
    fold_rows=[]
    for (dataset,fitness,fold),group in sorted(by_fold.items(),key=lambda kv:(kv[0][0],kv[0][1],int(kv[0][2]))):
        entry=dict(dataset=dataset,fitness=fitness,fold=fold,n_runs=len(group))
        for m in metric_keys:
            vals=[v for r in group if (v:=finite(r,m)) is not None]
            entry[f'{m}_mean'],entry[f'{m}_std']=stats(vals)
        fold_rows.append(entry)
    write_tsv(out/'report_folds.tsv',fold_cols,fold_rows)
    # Aggregate first over runs within each fold, then across folds (not over 100 pseudo-independent runs).
    summary_cols=['dataset','fitness','n_folds','n_runs','complete_folds']+[f'{m}_{s}' for m in metric_keys for s in ('mean','std')]
    summary=[]
    completeness=[]
    for dataset,fitness in all_keys:
        group=[r for r in fold_rows if (r['dataset'],r['fitness'])==(dataset,fitness)]
        num_runs=sum(int(r['n_runs']) for r in group)
        complete=sum(int(r['n_runs'])==args.runs for r in group)
        entry=dict(dataset=dataset,fitness=fitness,n_folds=len(group),n_runs=num_runs,complete_folds=complete)
        for m in metric_keys:
            vals=[v for r in group if (v:=finite(r,f'{m}_mean')) is not None]
            entry[f'{m}_mean'],entry[f'{m}_std']=stats(vals)
        summary.append(entry)
        missing=[]
        for fold in range(1,args.folds+1):
            seen={r['seed'] for r in ok if (r['dataset'],r['fitness'],r['fold'])==(dataset,fitness,str(fold))}
            # No assumption about seed_start: only report counts.
            if len(seen)!=args.runs:missing.append(f'{fold}:{len(seen)}/{args.runs}')
        completeness.append(dict(dataset=dataset,fitness=fitness,successful=num_runs,
                                 expected=args.folds*args.runs,complete_folds=complete,
                                 missing_by_fold=','.join(missing),failed_records=failures[(dataset,fitness)]))
    write_tsv(out/'report_summary.tsv',summary_cols,summary)
    write_tsv(out/'report_completeness.tsv',['dataset','fitness','successful','expected','complete_folds','missing_by_fold','failed_records'],completeness)
    print('Input:',args.input)
    print('Successful unique runs:',len(ok),'(duplicates removed:',sum(r.get('status')=='OK' for r in rows)-len(ok),')')
    for name in ('report_runs.tsv','report_folds.tsv','report_summary.tsv','report_completeness.tsv'):
        print('Created:',out/name)
    if not any('precision' in m for m in metric_keys):
        print('WARNING: Precision missing. First run enrich_metrics.py on the original results.tsv to recover it from confusion-matrix logs.',file=sys.stderr)
    for r in completeness:
        if r['successful']!=r['expected']:
            print(f"INCOMPLETE: {r['dataset']} {r['fitness']}: {r['successful']}/{r['expected']}",file=sys.stderr)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError) as e:sys.exit(f'ERROR: {e}')
