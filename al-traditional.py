import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import sys


if len(sys.argv) != 5:
    print("Usage:", sys.argv[0], "<query strategy> <seed size> <pool size> <num of iterations>", file = sys.stderr)
    print("<query strategy> = uncertainty|margin|entropy", file = sys.stderr)
    sys.exit(0)

if sys.argv[1] == "uncertainty":
    query_strategy = uncertainty_sampling
elif sys.argv[1] == "margin":
    query_strategy = margin_sampling
elif sys.argv[1] == "entropy":
    query_strategy = entropy_sampling
else:
    print("<query strategy> = uncertainty|margin|entropy", file = sys.stderr)
    sys.exit(0)

query_string = sys.argv[1]
seed_size = int(sys.argv[2])
pool_size = int(sys.argv[3])
iterations = int(sys.argv[4])

iter = 0
rand = 0

while iter < iterations:

    rng = np.random.default_rng(rand)

    training_set = pd.read_csv('/home/risto/al-training.csv')

    # build the seed

    idx = rng.choice(training_set.index, seed_size, replace=False)
    seed = training_set[training_set.index.isin(idx)]
    training_set = training_set.drop(idx)

    temp = seed[seed['Label'] == 0]

    if len(temp) == 0:
        print("Iteration:", iter+1, "no points with label 0 in seed, regenerating", file = sys.stderr)
        rand += 1
        continue

    temp = seed[seed['Label'] == 1]

    if len(temp) == 0:
        print("Iteration:", iter+1, "no points with label 1 in seed, regenerating", file = sys.stderr)
        rand += 1
        continue

    # build the pool

    idx = rng.choice(training_set.index, pool_size, replace=False)
    pool = training_set[training_set.index.isin(idx)]
    training_set = training_set.drop(idx)

    temp = pool[pool['Label'] == 0]

    if len(temp) == 0:
        print("Iteration:", iter+1, "no points with label 0 in pool, regenerating", file = sys.stderr)
        rand += 1
        continue

    temp = pool[pool['Label'] == 1]

    if len(temp) == 0:
        print("Iteration:", iter+1, "no points with label 1 in pool, regenerating", file = sys.stderr)
        rand += 1
        continue

    # do active learning

    X_seed = seed.drop(columns=['Timestamp', 'SignatureText', 'Label'])
    y_seed = seed['Label']

    X_pool = pool.drop(columns=['Timestamp', 'SignatureText', 'Label'])
    y_pool = pool['Label']

    clf = RandomForestClassifier(n_estimators=100, random_state=iter,
                                 class_weight='balanced')

    learner = ActiveLearner(
        estimator=clf,
        query_strategy=query_strategy,
        X_training=X_seed, y_training=y_seed.to_numpy()
    )

    test_set = pd.read_csv('/home/risto/al-test.csv')

    X_test = test_set.drop(columns=['Timestamp', 'SignatureText', 'Label'])
    y_test = test_set['Label']

    max_precision = 0.0
    max_recall = 0.0
    max_f1 = 0.0

    for i in range(1001):
        result = learner.predict(X_test)

        precision = precision_score(y_test, result)
        recall = recall_score(y_test, result)
        f1 = f1_score(y_test, result)

        print("Iteration:", iter+1, "instances:", i,
              "precision:", precision, "recall:", recall, "f1:", f1,
              file = sys.stderr, flush=True)

        if max_precision < precision:
            max_precision = precision
            max_precision_query = i

        if max_recall < recall:
            max_recall = recall
            max_recall_query = i

        if max_f1 < f1:
            max_f1 = f1
            max_f1_query = i

        if i == 1000:
            last_precision = precision
            last_recall = recall
            last_f1 = f1
            break

        # Note that modAL learner.query() method returns index as a row number,
        # while pandas dataframes use row numbers as string tags in their index. 
        # For example, calling drop() for the 3rd row will move the 4th row to 
        # third position, and the new 3rd row will now have the ID of '3' instead 
        # of '2'. For this reason, reset_index() method is called before
        # learner.query() and drop(), in order to reset the index to default: 
        # 0, 1, 2, 3, ...

        X_pool = X_pool.reset_index(drop=True)
        y_pool = y_pool.reset_index(drop=True)

        index, X_instance = learner.query(X_pool)
        y_instance = y_pool[index]

        learner.teach(X_instance, y_instance.to_numpy())

        X_pool = X_pool.drop(index)
        y_pool = y_pool.drop(index)

    # print summary results from the current iteration to standard output

    print("Iteration:", iter+1, 
          "LastPrecision:", last_precision,
          "LastRecall:", last_recall,
          "LastF1:", last_f1, 
          "MaxPrecision:", max_precision, "MaxPrecQuery:", max_precision_query, 
          "MaxRecall:", max_recall, "MaxRecallQuery:", max_recall_query, 
          "MaxF1:", max_f1, "MaxF1query:", max_f1_query, 
          flush=True);

    # increment the variable that resets the random number generator,
    # and also increment the loop variable

    rand += 1
    iter += 1
