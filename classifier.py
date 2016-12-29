import pandas as pd
import numpy as np

DF = pd.read_csv("features.csv", index_col=None, header=None)

from sklearn import svm
classifier = svm.SVC()
classifier.fit(DF.drop(["name","label"], axis=1), DF.label)
np.sum(classifier.predict(DF.drop(["name","label"], axis=1))== DF.label)