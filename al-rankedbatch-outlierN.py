import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import sys
from functools import partial


def mydistance(X, Y):

    if distance_type == 0:
        return 0.0

    if distance_type == 1:
        if X[0] != Y[0]:
            return 1.0
        else:
            return 0.0

    if distance_type == 2:
        return abs(X[8] - Y[8])

    return 0.0


if len(sys.argv) != 7:
    print("Usage:", sys.argv[0], "<batch size> <distance type> <outlier proportion> <seed size> <pool size> <num of iterations>", file = sys.stderr)
    sys.exit(0)

batch_size = int(sys.argv[1])
distance_type = int(sys.argv[2])
proportion = float(sys.argv[3])
seed_size = int(sys.argv[4])
pool_size = int(sys.argv[5])
iterations = int(sys.argv[6])

if proportion <= 0 or proportion > 1:
    print("<proportion> must be positive real number not greater than 1", 
          file = sys.stderr)
    sys.exit(0)

if int(proportion * seed_size) < 1:
    print("<proportion> is too small for the given seed size", 
          file = sys.stderr)
    sys.exit(0)

if int(proportion * pool_size) < 1:
    print("<proportion> is too small for the given pool size", 
          file = sys.stderr)
    sys.exit(0)

iter = 0
rand = 0

while iter < iterations:

    rng = np.random.default_rng(rand)

    training_set = pd.read_csv('/home/risto/al-training.csv')

    scas0 = training_set[training_set['SCAS'] == 0]
    scas1 = training_set[training_set['SCAS'] == 1]

    # build the seed

    idx = rng.choice(scas0.index, seed_size - int(proportion * seed_size) , replace=False)
    seed0 = scas0[scas0.index.isin(idx)]
    scas0 = scas0.drop(idx)

    idx = rng.choice(scas1.index, int(proportion * seed_size), replace=False)
    seed1 = scas1[scas1.index.isin(idx)]
    scas1 = scas1.drop(idx)

    seed = pd.concat([seed0, seed1], ignore_index=True)

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

    idx = rng.choice(scas0.index, pool_size - int(proportion * pool_size), replace=False)
    pool0 = scas0[scas0.index.isin(idx)]
    scas0 = scas0.drop(idx)

    idx = rng.choice(scas1.index, int(proportion * pool_size), replace=False)
    pool1 = scas1[scas1.index.isin(idx)]
    scas1 = scas1.drop(idx)

    pool = pd.concat([pool0, pool1], ignore_index=True)

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

    X_seed = seed.drop(columns=['Timestamp', 'SignatureText', 'Label']).to_numpy()
    y_seed = seed['Label'].to_numpy()

    X_pool = pool.drop(columns=['Timestamp', 'SignatureText', 'Label']).to_numpy()
    y_pool = pool['Label'].to_numpy()

    clf = RandomForestClassifier(n_estimators=100, random_state=iter,
                                 class_weight='balanced')

    preset_batch = partial(uncertainty_batch_sampling, 
                           n_instances=batch_size, 
                           metric=mydistance)

    learner = ActiveLearner(
        estimator=clf,
        query_strategy=preset_batch,
        X_training=X_seed, y_training=y_seed
    )

    test_set = pd.read_csv('/home/risto/al-test.csv')

    X_test = test_set.drop(columns=['Timestamp', 'SignatureText', 'Label']).to_numpy()
    y_test = test_set['Label'].to_numpy()

    max_precision = 0.0
    max_recall = 0.0
    max_f1 = 0.0

    for i in range(0, 1001, batch_size):

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

        index, _ = learner.query(X_pool)

        X_instances = X_pool[index]
        y_instances = y_pool[index]

        learner.teach(X=X_instances, y=y_instances)

        X_pool = np.delete(X_pool, index, axis=0)
        y_pool = np.delete(y_pool, index, axis=0)

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

