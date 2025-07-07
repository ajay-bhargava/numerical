import numpy as np
import typer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

app = typer.Typer()

def load_data():
    data = load_iris()
    X, y = data.data, data.target # type: ignore
    return X, y

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    return np.sqrt(np.sum((x1 - x2) ** 2))

def _knn(X, y, x, k, distance_fn):
    distances = [distance_fn(x, x_i) for x_i in X]
    k_nearest = np.argsort(distances)[:k]
    nearest_y = y[k_nearest]
    values, counts = np.unique(nearest_y, return_counts=True)
    return values[np.argmax(counts)]

@app.command(
    help="K-Nearest Neighbors"
)
def knn(
    k: int = typer.Option(3, "--k", "-k", help="Number of nearest neighbors"),
    distance_fn: str = typer.Option("euclidean", "--distance-fn", "-d", help="Distance function"),
):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    y_pred = [_knn(X_train, y_train, x, k, _euclidean_distance) for x in X_test]
    
    accuracy = sum(1 for pred, actual in zip(y_pred, y_test) if pred == actual) / len(y_test)
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    app()
    
