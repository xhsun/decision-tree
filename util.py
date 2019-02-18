import numpy as np


def partition_classes(X, y, split_attribute, split_val):
    """
    Example:

    X = [[3, 'aa', 10],                 y = [1,
         [1, 'bb', 22],                      1,
         [2, 'cc', 28],                      0,
         [5, 'bb', 32],                      0,
         [4, 'cc', 32]]                      1]

    Here, columns 0 and 2 represent numeric attributes, while column 1 is a categorical attribute.

    Consider the case where we call the function with split_attribute = 0 and split_val = 3 (mean of column 0)
    Then we divide X into two lists - X_left, where column 0 is <= 3  and X_right, where column 0 is > 3.

    X_left = [[3, 'aa', 10],                 y_left = [1,
              [1, 'bb', 22],                           1,
              [2, 'cc', 28]]                           0]

    X_right = [[5, 'bb', 32],                y_right = [0,
               [4, 'cc', 32]]                           1]

    Consider another case where we call the function with split_attribute = 1 and split_val = 'bb'
    Then we divide X into two lists, one where column 1 is 'bb', and the other where it is not 'bb'.

    X_left = [[1, 'bb', 22],                 y_left = [1,
              [5, 'bb', 32]]                           0]

    X_right = [[3, 'aa', 10],                y_right = [1,
               [2, 'cc', 28],                           0,
               [4, 'cc', 32]]                           1]

    :param X: data containing all attributes
    :param y: labels
    :param split_attribute: column index of the attribute to split on
    :param split_val: either a numerical or categorical value to divide the split_attribute
    :return: The partitioned X and Y
    """

    X_left = []
    X_right = []

    y_left = []
    y_right = []

    is_string = isinstance(split_val, str)

    for i in range(len(X)):
        x = X[i]
        label = y[i]
        attr = x[split_attribute]
        if is_string:
            if attr == split_val:
                X_left.append(x)
                y_left.append(label)
            else:
                X_right.append(x)
                y_right.append(label)
        else:
            if attr <= split_val:
                X_left.append(x)
                y_left.append(label)
            else:
                X_right.append(x)
                y_right.append(label)

    return X_left, X_right, y_left, y_right


def information_gain(previous_y, current_y):
    """
    Example:

    previous_y = [0,0,0,1,1,1]
    current_y = [[0,0], [1,1,1,0]]

    info_gain = 0.45915

    :param previous_y: the distribution of original labels (0's and 1's)
    :param current_y: the distribution of labels after splitting based on a particular split attribute and split value
    :return: information gain for previous y and current y
    """

    parent_entropy = entropy(previous_y)
    parent_len = len(previous_y)

    child_entropy = 0
    for child in current_y:
        child_entropy += entropy(child) * (len(child) / parent_len)

    info_gain = parent_entropy - child_entropy

    return info_gain


#
def entropy(class_y):
    """
    Example:

    entropy([0,0,0,1,1,1,1,1,1]) = 0.92

    This method computes entropy for information gain
    :param class_y: list of class labels (0's and 1's)
    :return: entropy for the given y
    """

    unique, counts = np.unique(class_y, return_counts=True)
    probabilities = counts / len(class_y)

    ent = 0
    for p in probabilities:
        ent -= p * np.log2(p)

    return ent
