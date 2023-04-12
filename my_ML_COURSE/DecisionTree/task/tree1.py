import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    n_objects, n_classes = y.shape
    counts = np.sum(y, axis=0)
    probs = counts / n_objects
    entropy = -np.sum(probs * np.log2(probs + EPS))
    # return entropy

    return 0.0


def gini(y):
    """
    Computes the Gini impurity of the provided distribution

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset

    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    n_objects, n_classes = y.shape
    counts = np.sum(y, axis=0)
    probs = counts / n_objects
    gini_impurity = 1 - np.sum(probs**2)
    # return gini_impurity

    return 0.0


def variance(y):
    """
    Computes the variance the provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Variance of the provided target vector
    """

    # YOUR CODE HERE
    mean_y = np.mean(y)
    variance = np.mean((y - mean_y) ** 2)

    # return variance

    return 0.0


def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset

    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector

    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    median_y = np.median(y)
    mad_median = np.mean(np.abs(y - median_y))
    # return mad_median

    return 0.0


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.0 
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """

    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None


class DecisionTree(BaseEstimator):
    all_criterions = {
        "gini": (gini, True),  # (criterion, classification flag)
        "entropy": (entropy, True),
        "variance": (variance, False),
        "mad_median": (mad_median, False),
    }

    def __init__(
        self,
        n_classes=None,
        max_depth=np.inf,
        min_samples_split=2,
        criterion_name="gini",
        debug=False,
    ):
        assert (
            criterion_name in self.all_criterions.keys()
        ), "Criterion name must be on of the following: {}".format(
            self.all_criterions.keys()
        )

        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None  # Use the Node class to initialize it later
        self.debug = debug

    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE

        left_indices = np.where(X_subset[:, feature_index] < threshold)[0]
        right_indices = np.where(X_subset[:, feature_index] >= threshold)[0]
        X_left, y_left = X_subset[left_indices], y_subset[left_indices]
        X_right, y_right = X_subset[right_indices], y_subset[right_indices]

        return (X_left, y_left), (X_right, y_right)

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold

        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels for corresponding subset

        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification
                   (n_objects, 1) in regression
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        y_left = y_subset[X_subset[:, feature_index] < threshold]
        y_right = y_subset[X_subset[:, feature_index] >= threshold]

        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        best_feature, best_threshold, best_score = None, None, -np.inf

        for feature_index in range(X_subset.shape[1]):
            feature_values = np.unique(X_subset[:, feature_index])
            for threshold in feature_values:

                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)

                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                clf_error_left = self.criterion(y_left)
                clf_error_right = self.criterion(y_right)

                score = (len(y_left) * clf_error_left + len(y_right) * clf_error_right) / len(y_subset)

                if score > best_score:
                    best_feature, best_threshold, best_score = feature_index, threshold, score




        return feature_index, threshold

    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in
            classification, (n_objects, 1) in regression One-hot representation
            of class labels or target values for corresponding subset.

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold, proba=0)

        return new_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), "Wrong y shape"
        self.criterion, self.classification = self.all_criterions[self.criterion_name]

        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification
                   (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """

        # YOUR CODE HERE

        # Make predictions using the decision tree
        y_predicted = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            node = self.root
            while not node.is_leaf():
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
            y_predicted[i] = node.predicted_class if self.problem_type == "classification" else node.predicted_value

        return y_predicted

    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects

        """
        assert self.classification, "Available only for classification problem"

        # YOUR CODE HERE

        return y_predicted_probs
