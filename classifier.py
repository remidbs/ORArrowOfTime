import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score

DF = pd.read_csv("features/features30-12.csv", index_col=None, header=None)
DF = DF.rename(columns={len(DF.columns)-1 : "label",len(DF.columns)-2:"name"})
classifier = svm.SVC()

n_splits = 4
scores = cross_val_score(classifier, DF.drop(["name","label"], axis=1), DF.label, cv=4)
print "Average prediction score for ",n_splits," splits : ",np.mean(scores)