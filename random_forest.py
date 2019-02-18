import multiprocessing
import random

import numpy as np

from decision_tree import DecisionTree


class RandomForest(object):
    """
    1. X is assumed to be a matrix with n rows and d columns where n is the
    number of total records and d is the number of features of each record.
    2. y is assumed to be a vector of labels of length n.
    3. XX is similar to X, except that XX also contains the data label for each
    record.
    """
    num_trees = 0
    decision_trees = []

    bootstraps_datasets = []
    bootstraps_labels = []

    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.decision_trees = [DecisionTree() for i in range(num_trees)]

    def bootstrapping(self, XX):
        """
        Bootsrapping original databset for each tree
        :param XX: original dataset
        """
        for i in range(self.num_trees):
            data_sample, data_label = self._bootstrapping(XX, len(XX))
            self.bootstraps_datasets.append(data_sample)
            self.bootstraps_labels.append(data_label)

    def _bootstrapping(self, XX, n):
        """
        Create a sample dataset of size n by sampling with replacement from the original dataset XX.
        :param XX: Original dataset
        :param n: sample dataset size
        :return: sample dataset
        """
        samples = []  # sampled dataset
        labels = []  # class labels for the sampled records

        # Randomly choose n indices and get data
        indices = np.random.choice(list(range(len(XX))), n)
        for i in indices:
            data = XX[i]
            samples.append(data[:-1])
            labels.append(data[-1])

        return samples, labels

    def fitting(self):
        """
        Train `num_trees` decision trees using the bootstraps datasets and labels by calling the learn function
        of the decision tree.
        """
        # Fitting trees in parallel
        pool = multiprocessing.Pool(4)
        pool.map(self._learn, (i for i in range(self.num_trees)))
        pool.close()
        pool.join()

        # Fitting trees sequentially
        # [self._learn(i) for i in range(self.num_trees)]

    def _learn(self, i):
        """
        Execute learning method for the decision tree at index i
        :param i: Index of the decision tree
        """
        self.decision_trees[i].learn(self.bootstraps_datasets[i], self.bootstraps_labels[i])

    def voting(self, X):
        """
        Predict the given dataset
        :param X: A matrix with n rows and d columns
        :return: Prediction of the random forest
        """
        y = []

        for record in X:
            votes = []
            for i in range(len(self.bootstraps_datasets)):
                dataset = self.bootstraps_datasets[i]
                # Find the tree that consider the record as an out-of-bag sample
                if record not in dataset:
                    # Predict the label using each of the above found trees
                    OOB_tree = self.decision_trees[i]
                    effective_vote = OOB_tree.classify(record)
                    votes.append(effective_vote)

            # Use majority vote to find the final label for this record
            counts = np.bincount(votes)

            if len(counts) == 0:
                # the record is not an out-of-bag sample for any of the trees
                y = np.append(y, random.choice([0, 1]))
            else:
                y = np.append(y, np.argmax(counts))

        return y
