import numpy as np


class KNN:
    def __init__(self, X_train, y_train, k=3):
        self.X_train = X_train
        self.y_train = y_train
        if k >=0 and k <= len(y_train) and k % 2 != 0:
            self.k = k
        else:
            raise ValueError("k must be a positive odd integer less than or equal to the number of training samples.")


    def predict(self, x_test):
        predicitons = []
        for x in x_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            near_k = np.argsort(distances)[:self.k]
            k_near_laber = self.y_train[near_k]
            values, counts = np.unique(k_near_laber, return_counts=True)
            predicitons.append(values[np.argmax(counts)])
        return np.array(predicitons)
