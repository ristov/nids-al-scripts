import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling, consensus_entropy_sampling, max_disagreement_sampling

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import sys


if len(sys.argv) != 7:
    print("Usage:", sys.argv[0], "<query strategy> <committee size> <outlier proportion> <seed size> <pool size> <num of iterations>", file = sys.stderr)
    print("<query strategy> = vote|consensus|disagreement", file = sys.stderr)
    sys.exit(0)

if sys.argv[1] == "vote":
    query_strategy = vote_entropy_sampling
elif sys.argv[1] == "consensus":
    query_strategy = consensus_entropy_sampling
elif sys.argv[1] == "disagreement":
    query_strategy = max_disagreement_sampling
else:
    print("<query strategy> = vote|consensus|disagreement", file = sys.stderr)
    sys.exit(0)

query_string = sys.argv[1]
committee_size = int(sys.argv[2])
proportion = float(sys.argv[3])
seed_size = int(sys.argv[4])
pool_size = int(sys.argv[5])
iterations = int(sys.argv[6])

if committee_size < 2:
    print("<committee size> must be at least 2", file = sys.stderr)
    sys.exit(0)


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

    # build the seed and create the committee

    learner_list = list()

    committee_failure = False

    for i in range(committee_size):

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
            committee_failure = True
            break

        temp = seed[seed['Label'] == 1]

        if len(temp) == 0:
            print("Iteration:", iter+1, "no points with label 1 in seed, regenerating", file = sys.stderr)
            committee_failure = True
            break

        X_seed = seed.drop(columns=['Timestamp', 'SignatureText', 'Label'])
        y_seed = seed['Label']

        clf = RandomForestClassifier(n_estimators=100, 
                                     random_state=iter*committee_size + i,
                                     class_weight='balanced')
    
        learner = ActiveLearner(
            estimator=clf,
            X_training=X_seed, y_training=y_seed.to_numpy()
        )
    
        learner_list.append(learner)

    if committee_failure:
        rand += 1
        continue

    committee = Committee(learner_list=learner_list,
                          query_strategy=query_strategy)

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

    X_pool = pool.drop(columns=['Timestamp', 'SignatureText', 'Label'])
    y_pool = pool['Label']

    # do active learning

    test_set = pd.read_csv('/home/risto/al-test.csv')

    X_test = test_set.drop(columns=['Timestamp', 'SignatureText', 'Label'])
    y_test = test_set['Label']

    max_precision = 0.0
    max_recall = 0.0
    max_f1 = 0.0

    for i in range(1001):
        result = committee.predict(X_test)

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

        index, X_instance = committee.query(X_pool)
        y_instance = y_pool[index]

        committee.teach(X_instance, y_instance.to_numpy())

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

