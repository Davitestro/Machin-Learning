from NN import NeuralNetwork, DenseLayer
import numpy as np

def generate_nn_friendly_dataset(
    n_samples=1000,
    n_features=2,
    noise=0.1,
    radius=1.0,
    seed=42
):
    np.random.seed(seed)

    X = np.random.uniform(-1.5, 1.5, size=(n_samples, n_features))

    dist = np.linalg.norm(X, axis=1)

    y = (dist < radius).astype(int)

    X += np.random.normal(0, noise, size=X.shape)

    return X, y


def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

x, y = generate_nn_friendly_dataset(n_samples=5000, noise=0.1, n_features=6, radius=1.0, seed=42)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, seed=42)
y_train = y_train.reshape(-1, 1)


NerN = NeuralNetwork()
NerN.add(DenseLayer(6, 16, activation='relu', lr=0.05))
NerN.add(DenseLayer(16, 16, activation='relu', lr=0.05))
NerN.add(DenseLayer(16, 1, activation='sigmoid', lr=0.05))

NerN.fit(x_train, y_train, epochs=200)
y_pred = NerN.predict(x_test)
y_pred = y_pred.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


print("accuracy:", np.mean((y_pred > 0.5) == y_test))