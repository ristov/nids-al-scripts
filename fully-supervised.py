import pandas as pd
import numpy as np

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import sys


if len(sys.argv) != 2:
    print("Usage:", sys.argv[0], "<num of iterations>", file = sys.stderr)
    sys.exit(0)

iterations = int(sys.argv[1])


training_set = pd.read_csv('/home/risto/al-training.csv')

X_train = training_set.drop(columns=['Timestamp', 'SignatureText', 'Label'])
y_train = training_set['Label']

label0 = y_train[y_train == 0]
label1 = y_train[y_train == 1]


test_set = pd.read_csv('/home/risto/al-test.csv')

X_test = test_set.drop(columns=['Timestamp', 'SignatureText', 'Label'])
y_test = test_set['Label']

label0 = y_test[y_test == 0]
label1 = y_test[y_test == 1]


for i in range(iterations):
    
    clf = RandomForestClassifier(n_estimators=100, random_state=i, class_weight='balanced')
    
    model = clf.fit(X_train, y_train)
    
    result = model.predict(X_test)
    
    precision = precision_score(y_test, result)
    recall = recall_score(y_test, result)
    f1 = f1_score(y_test, result)

    print("IterationRFweighted:", i+1, 
          "Precision:", precision,
          "Recall:", recall,
          "F1:", f1, 
          flush=True);


for i in range(iterations):
    
    ros = RandomOverSampler(random_state=i)
    
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=i)
    
    model = clf.fit(X_resampled, y_resampled)
    
    result = model.predict(X_test)
    
    precision = precision_score(y_test, result)
    recall = recall_score(y_test, result)
    f1 = f1_score(y_test, result)
    
    print("IterationRFoversampling:", i+1, 
          "Precision:", precision,
          "Recall:", recall,
          "F1:", f1, 
          flush=True);


for i in range(iterations):
    
    rus = RandomUnderSampler(random_state=i)
    
    X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

    label0 = y_resampled[y_resampled == 0]
    label1 = y_resampled[y_resampled == 1]

    clf = RandomForestClassifier(n_estimators=100, random_state=i)
    
    model = clf.fit(X_resampled, y_resampled)
    
    result = model.predict(X_test)
    
    precision = precision_score(y_test, result)
    recall = recall_score(y_test, result)
    f1 = f1_score(y_test, result)
    
    print("IterationRFundersampling:", i+1, 
          "Precision:", precision,
          "Recall:", recall,
          "F1:", f1, 
          flush=True);

