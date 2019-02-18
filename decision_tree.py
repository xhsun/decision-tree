import numpy as np

from util import information_gain, partition_classes


class DecisionTree(object):
    def __init__(self):
        # Initializing the tree as an empty dictionary
        self.tree = {}
        self.default = 0

    def learn(self, X, y):
        # Train the decision tree (self.tree) using the the sample X and labels y
        if len(X) > 0:
            unique, counts = np.unique(y, return_counts=True)
            index = np.argmax(counts)
            self.default = unique[index]
            self.tree = self._split(X, y, list(range(len(X[0]))))

    def classify(self, record):
        # classify the record using self.tree and return the predicted label
        attributes = list(range(len(record)))
        return self._predict(record, self.tree, attributes) or 0

    def _predict(self, record, tree, attributes):
        # Try to predict outcome for the given record
        for i in attributes:
            if i in list(tree.keys()):
                value = record[i]
                if isinstance(value, str):
                    return self.__predict(record, tree, attributes, i, value)
                else:
                    subtree = tree[i]
                    if isinstance(subtree, dict):
                        num_str = list(subtree.keys())[0][1:]
                        num = float(num_str)
                        if value <= num:
                            key = "<" + num_str
                        else:
                            key = ">" + num_str
                        return self.__predict(record, tree, attributes, i, key)
                    else:
                        return subtree

    def _split(self, X, y, attributes):
        # Create the tree base on provided values
        attributes = attributes.copy()
        if len(attributes) < 1:
            if len(X) <= 0:
                return self.default
            else:
                index = np.bincount(y).argmax()
                return y[index]
        elif len(X) <= 1:
            return self.default
        else:
            target, splits = self.__choose_attribute(X, y, attributes)
            attributes.remove(target)
            tree = {target: {}}

            for split, result in splits.items():
                x_list = result[0]
                y_list = result[1]
                if len(x_list) == 1:
                    self.__branch_categorical(x_list, y_list, attributes, target, split, tree)
                else:
                    self.__branch_numeric(x_list, y_list, attributes, target, split, tree)
            return tree

    def __predict(self, record, tree, attributes, attribute, split):
        # Try to find next node in the tree and keep going until find a leaf
        try:
            result = tree[attribute][split]
        except KeyError:
            return self.default
        if isinstance(result, dict):
            return self._predict(record, result, attributes)
        else:
            return result

    def __branch_categorical(self, x_list, y_list, attributes, target, split, tree):
        # Create subtree for categorical split
        self.___branch(x_list[0], y_list[0], attributes, target, split, tree)

    def __branch_numeric(self, x_list, y_list, attributes, target, split, tree):
        # Create subtree for numeric split
        xl = x_list[0]
        xr = x_list[1]
        yl = y_list[0]
        yr = y_list[1]

        self.___branch(xl, yl, attributes, target, "<" + str(split), tree)
        self.___branch(xr, yr, attributes, target, ">" + str(split), tree)

    def __choose_attribute(self, X, y, attributes):
        # Find the best attribute that offers max information gain
        first_run = True
        best = attributes[0]
        max_gain = 0
        splits = {}

        for a in attributes:
            gain, temp_splits = self.___calculate_gain(X, y, a, [row[a] for row in X])

            if first_run:  # To make sure this method always provides an attribute
                max_gain = gain
                best = a
                splits = temp_splits
                first_run = False

            if gain > max_gain:
                max_gain = gain
                best = a
                splits = temp_splits
        return best, splits

    def ___branch(self, X, y, attributes, target, split, tree):
        # Create subtree for the given split
        if len(X) <= 0:
            tree[target][split] = self.default
        if len(X) == 1:
            tree[target][split] = y[0] or 0
        else:
            subtree = self._split(X, y, attributes)
            tree[target][split] = subtree

    def ___calculate_gain(self, X, y, attribute, values):
        # Calculate information gain for given attribute
        splits = {}
        if isinstance(values[0], str):
            targets = np.unique(values)
            new_y = []
            for s in targets:
                xl, _, yl, _ = partition_classes(X, y, attribute, s)
                splits[s] = [[xl], [yl]]
                new_y.append(yl)
            gain = information_gain(y, new_y)
        else:
            s = np.mean(values)
            xl, xr, yl, yr = partition_classes(X, y, attribute, s)
            splits[s] = [[xl, xr], [yl, yr]]
            gain = information_gain(y, [yl, yr])
        return gain, splits
