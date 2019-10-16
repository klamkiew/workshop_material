#!/usr/bin/env python3

"""
Takes two different data sets (positive and negative) as input and
trains the a random forest classifier based on these training sets.

Dependencies:
numpy
scikit-learn

Author:
Kevin Lamkiewicz

USAGE:
python3 rf_classifier.py <POSITIVE_SET> <NEGATIVE_SET>
"""

import sys
import itertools
import random
from collections import Counter

import numpy as np

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split


def read_training_set(positive_set, negative_set):
    """
    This method reads the two training sets
    and replaces U's with T's. This is mainly done
    because all NCBI genomes (even RNA virus genomes)
    have T's in their sequence.

    Keyword arguments:
    positive_set -- fasta file with positive training instances
    negative_set -- fasta file with negative training instances
    """

    sequences = []
    # read the positive set
    with open(positive_set, 'r') as input_stream:
        for line in itertools.islice(input_stream, 1, None, 2):
            # each sequence gets the miRNA flag for classification
            sequences.append((line.upper().replace('T', 'U').rstrip('\n'), 'miRNA'))

    # read the negative set
    with open(negative_set, 'r') as input_stream:
        for line in itertools.islice(input_stream, 1, None, 2):
            # each sequence gets the pseudo flag for classification
            sequences.append((line.upper().replace('T', 'U').rstrip('\n'), 'pseudo'))
    return(sequences)

def transform_data(data):
    """
    This function transforms each sequence in data
    into a multi-dimensional vector which can be
    used by scikit-learn to train a machine 
    learning model.

    Keyword arguments:
    data -- an array with 2-tuples, containing each sequence and their respective class
    """ 

    trainingSet = []
    targets = []

    for dataPoint in data:
        group = dataPoint[1]
        sequence = dataPoint[0]
        length = len(sequence)
        relGC = float((Counter(sequence)['C'] + Counter(sequence)['G']) / length)
        freqA = float(Counter(sequence)['A'] / length)
        freqC = float(Counter(sequence)['C'] / length)
        freqG = float(Counter(sequence)['G'] / length)
        freqU = float(Counter(sequence)['U'] / length)
        
        vector = [length, relGC, freqA, freqC, freqG, freqU]

        trainingSet.append(vector)
        targets.append(group)

    return(np.array(trainingSet, dtype=float), targets)

if __name__ == '__main__':
    """
    Main Routine
    This block of code is executed, whenever the script
    is started from the command line.
    """

    print("Starting the Script")
    positive = sys.argv[1]
    negative = sys.argv[2]

    trainingsData = read_training_set(positive, negative)

    # it is good practice to shuffle your data
    random.shuffle(trainingsData)

    print("Loaded trainings data")

    # let's have a look at the first five
    # sequences and their structure
    # print(trainingsData[:5])
    # sys.exit(0)

    trainingSet, targets = transform_data(trainingsData)
    print('Data has been transformed into a numpy array')

    # again, let us look at the first two instances
    # print(trainingSet[:5])
    # sys.exit(0)

    # we know this already. we split the data into training and test sets
    data_training, data_test, target_training, target_test = train_test_split(trainingSet, targets, test_size=0.2)

    # creating the random forest classifier and already fitting the training data
    rfc = RandomForestClassifier(n_estimators=100, max_depth=2).fit(data_training, target_training)
    # prediction of the test set
    prediction = rfc.predict(data_test)
    # output the confusion matrix to evaluate the model
    print()
    print(metrics.confusion_matrix(target_test, prediction))
    print(metrics.accuracy_score(target_test, prediction))
        