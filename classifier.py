import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split

DF = pd.read_csv("features/features9-1_all_fc8_20.csv", index_col=None, header=None)
DF = DF.rename(columns={len(DF.columns)-1 : "label",len(DF.columns)-2:"name"})

DF_forward = DF[DF.label]
F_X_AB, F_X_C, F_y_AB, F_y_C = train_test_split(DF_forward.drop(["name","label"], axis=1),DF_forward.label, test_size = 1.0/3)
F_X_A, F_X_B, F_y_A, F_y_B = train_test_split(F_X_AB, F_y_AB, test_size= 0.5 )

DF_backward = DF[DF.label == False]
B_X_CB, B_X_A, B_y_CB, B_y_A = train_test_split(DF_backward.drop(["name","label"], axis=1),DF_backward.label, test_size = 1.0/3)
B_X_C, B_X_B, B_y_C, B_y_B = train_test_split(B_X_CB, B_y_CB, test_size= 0.5 )

X_A = pd.concat([B_X_A,F_X_A], axis=0)
y_A = pd.concat([B_y_A,F_y_A], axis=0)
X_B = pd.concat([B_X_B,F_X_B], axis=0)
y_B = pd.concat([B_y_B,F_y_B], axis=0)
X_C = pd.concat([B_X_C,F_X_C], axis=0)
y_C = pd.concat([B_y_C,F_y_C], axis=0)

for C in [0.01,0.1,1.0,10.0,100.0,1000.0,10000.0]:
    print "C:",C
    for name,X_1,y_1,X_2,y_2,X_3,y_3 in [("C",X_A,y_A,X_B,y_B,X_C,y_C),
                                         ("A",X_B,y_B,X_C,y_C,X_A,y_A),
                                         ("B",X_C,y_C,X_A,y_A,X_B,y_B)]:
        print "Testing on",name,". Training on the 2 other folds."
        X_train = pd.concat([X_1,X_2],axis=0)
        y_train = pd.concat([y_1,y_2],axis=0)
        X_test = X_3
        y_test = y_3
        classifier = svm.SVC(C, class_weight={True:0.16502,False:1})
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)
        print np.mean(1-np.abs(pred-y_test))
    

#%% 
from sklearn.ensemble import RandomForestClassifier as RF

classifier = RF(30,max_features=0.1, min_samples_leaf=20)
classifier.fit(DF.drop(["name","label"], axis=1), DF.label)
pred = classifier.predict(DF.drop(["name","label"], axis=1))
np.max(classifier.feature_importances_)
print np.mean(1-np.abs(pred-DF.label))
n_splits = 4
scores = cross_val_score(classifier, DF.drop(["name","label"], axis=1), DF.label, cv=4)
print "Average prediction score for ",n_splits," splits : ",np.mean(scores)
#%%
from matplotlib import pyplot as plt
import seaborn as sb
DF1 = DF[DF.label==True]
DF2 = DF[DF.label==False]
ft1 = 6
ft2 = 0
#plt.scatter(DF1[ft1],DF1[ft2], color="r")
#plt.scatter(DF2[ft1],DF2[ft2], color="g")
plt.scatter(DF1[ft1],np.zeros(DF1.shape[0]), color="r")
plt.scatter(DF2[ft1],np.zeros(DF2.shape[0]), color="g")

