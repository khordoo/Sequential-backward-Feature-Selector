"""
Created on Sat May 9 11:35:15 2020
@author: Mahmood Khordoo
https://github.com/khordoo
"""
from itertools import combinations
from sklearn.model_selection import cross_val_score


class SequentialBackwardSelector:
    """
    Sequential backward selector for Scikit-learn estimators.
    """

    def __init__(self, estimator, reduced_feature_size=1, use_cross_val=True):
        self.estimator = estimator
        self.use_cross_val = use_cross_val
        self.reduced_feature_size = reduced_feature_size
        self.best_features_ = []

    def fit(self, X, y):
        """Finds a reduced set of features that results in the highest accuracy.
        Sequentially removes the available features, evaluates the model
        accuracy the all the possible combinations
        of the reduced feature subspace and keeps the best(highest accuracy)
        combination of features for each reduced feature size.

        Parameters
        ----------
        X : Numpy array
        y : Numpy array

        Returns
        -------
         None
        """

        feature_size = X.shape[1]
        keep_features = range(feature_size)

        while feature_size >= self.reduced_feature_size:
            best_feature_combination = None
            best_score = 0

            for feature_combination in combinations(keep_features, feature_size):
                score = self._score(X, y, feature_combination)
                if score > best_score:
                    best_score = score
                    best_feature_combination = feature_combination

            keep_features = best_feature_combination
            self._save_score(feature_size, best_score, best_feature_combination)
            feature_size -= 1

    def _score(self, X, y, selected_feature_indexes):
        if self.use_cross_val:
            score = cross_val_score(self.estimator, X[:, selected_feature_indexes], y ).mean()
        else:
            self.estimator.fit(X[:, selected_feature_indexes], y)
            score = self.estimator.score(X[:, selected_feature_indexes], y)
        return score

    def _save_score(self, num_features, best_score, best_feature_combination):
        self.best_features_.append({
            'featureSize': num_features,
            'score': best_score,
            'features': best_feature_combination
        })
