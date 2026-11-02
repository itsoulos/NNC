#!/usr/bin/env python3
"""Independent reference metrics on hand-checked confusion matrices (stdlib only)."""
import math
import unittest

def metrics(cm):
    n=len(cm); total=sum(map(sum,cm)); assert total>0
    actual=[sum(row) for row in cm]
    predicted=[sum(cm[i][j] for i in range(n)) for j in range(n)]
    correct=sum(cm[i][i] for i in range(n))
    recall=[cm[i][i]/actual[i] if actual[i] else 0 for i in range(n)]
    precision=[cm[i][i]/predicted[i] if predicted[i] else 0 for i in range(n)]
    f1=[2*p*r/(p+r) if p+r else 0 for p,r in zip(precision,recall)]
    present=[i for i,a in enumerate(actual) if a]
    mr=sum(recall[i] for i in present)/len(present)
    mf=sum(f1[i] for i in present)/len(present)
    wf=sum(actual[i]*f1[i] for i in range(n))/total
    gm=math.prod(recall[i] for i in present)**(1/len(present))
    den=math.sqrt((total**2-sum(p*p for p in predicted))*(total**2-sum(a*a for a in actual)))
    mcc=(correct*total-sum(a*p for a,p in zip(actual,predicted)))/den if den else 0
    return dict(error=correct/total-1,balanced_error=mr-1,macro_recall=mr,macro_f1=mf,weighted_f1=wf,gmean=gm,mcc=mcc)

class MetricTests(unittest.TestCase):
    def test_perfect(self):
        m=metrics([[8,0],[0,2]])
        for k in ('macro_recall','macro_f1','weighted_f1','gmean','mcc'): self.assertAlmostEqual(m[k],1)
        self.assertAlmostEqual(m['error'],0)
    def test_majority_only(self):
        m=metrics([[8,0],[2,0]])
        self.assertAlmostEqual(m['macro_recall'],.5)
        self.assertAlmostEqual(m['macro_f1'],4/9)
        self.assertAlmostEqual(m['weighted_f1'],32/45)
        self.assertEqual(m['gmean'],0)
        self.assertEqual(m['mcc'],0)
    def test_imbalanced(self):
        m=metrics([[8,1],[1,1]])
        self.assertAlmostEqual(m['macro_recall'],(8/9+1/2)/2)
        self.assertAlmostEqual(m['macro_f1'],(8/9+1/2)/2)
        self.assertAlmostEqual(m['mcc'],7/18)
    def test_multiclass(self):
        m=metrics([[3,0,0],[1,1,0],[0,1,2]])
        self.assertAlmostEqual(m['macro_recall'],(1+.5+2/3)/3)
        self.assertGreaterEqual(m['gmean'],0)

if __name__=='__main__': unittest.main(verbosity=2)
