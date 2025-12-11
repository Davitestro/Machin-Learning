import numpy as np
from collections import Counter


class Tree:
    def __init__(self, x, y, depth=0, max_depth=5):
        self.x = x
        self.y = y
        self.depth = depth
        self.max_depth = max_depth
        self.is_leaf = False

        if len(set(y)) == 1 or self.depth == self.max_depth:
            self.is_leaf = True
            self.prediction = max(set(y), key=list(y).count)
            return

    def entropy(self, y):
        counter = Counter(y)
        total = len(y)
        ent = 0.0
        for count in counter.values():
            p = count / total
            ent -= p * np.log2(p)
        return ent

    def info_gain(self, left_y, right_y, parent_entropy):
        total = len(left_y) + len(right_y)
        p_left = len(left_y) / total
        p_right = len(right_y) / total
        child_entropy = p_left * self.entropy(left_y) + p_right * self.entropy(right_y)
        return parent_entropy - child_entropy
    
    def fit(self):
        best_gain = 0
        best_split = None
        parent_entropy = self.entropy(self.y)
        _, n_features = self.x.shape

        for feature_index in range(n_features):
            thresholds = np.unique(self.x[:, feature_index])
            for threshold in thresholds:
                left_indices = self.x[:, feature_index] <= threshold
                right_indices = self.x[:, feature_index] > threshold
                if len(self.y[left_indices]) == 0 or len(self.y[right_indices]) == 0:
                    continue

                gain = self.info_gain(self.y[left_indices], self.y[right_indices], parent_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }

        if best_gain > 0:
            self.feature_index = best_split['feature_index']
            self.threshold = best_split['threshold']

            self.left = Tree(
                self.x[best_split['left_indices']],
                self.y[best_split['left_indices']],
                depth=self.depth + 1,
                max_depth=self.max_depth
            )
            self.right = Tree(
                self.x[best_split['right_indices']],
                self.y[best_split['right_indices']],
                depth=self.depth + 1,
                max_depth=self.max_depth
            )
            self.left.fit()
            self.right.fit()
            return
        else:
            self.is_leaf = True
            self.prediction = Counter(self.y).most_common(1)[0][0]
        

    def _predict_one(self, x):
        if self.is_leaf:
            return self.prediction
        if x[self.feature_index] <= self.threshold:
            return self.left._predict_one(x)
        else:
            return self.right._predict_one(x)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

