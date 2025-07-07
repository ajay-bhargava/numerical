import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import typer

app = typer.Typer()

def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target # type: ignore
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def initialize_weights(X):
    return np.random.randn(X.shape[1]) * 0.01

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def logit(X, w, b):
    z = X @ w + b
    return sigmoid(z) # type: ignore

def loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient(X, y, w, b):
    y_pred = logit(X, w, b)
    dz = y_pred - y
    dw = X.T @ dz
    db = np.mean(dz)
    return dw, db

def accuracy(y_pred, y_true):
    return np.round(np.mean(y_pred == y_true), 2)

def predict_label(y_pred, threshold: float = 0.5):
    return (y_pred >= threshold).astype(int)

@app.command()
def logits(
    epochs: int = typer.Option(1000, "--epochs", "-e", help="Number of epochs"),
):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    w = initialize_weights(X_train)
    b = 0
    for i in range(epochs):
        y_pred = logit(X_train, w, b)
        loss_value = loss(y_pred, y_train)
        dw, db = gradient(X_train, y_train, w, b)
        w = w - 0.01 * dw
        b = b - 0.01 * db
        if i % 100 == 0:
            print(f"Epoch {i}, Loss: {loss_value}, Accuracy: {accuracy(predict_label(y_pred), y_train)}")
    y_pred = logit(X_test, w, b)
    print(f"Loss: {loss(y_pred, y_test)}")
    print(f"Accuracy: {accuracy(predict_label(y_pred), y_test)}")

if __name__ == "__main__":
    app()

