import numpy as np
from sklearn.datasets import load_iris
from typer import Typer
from rich import print
import typer

app = Typer()

iris = load_iris()

def plot_pca(X_pca, y, X_original):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('[bold red]matplotlib not installed. Skipping visualization.[/bold red]')
        return
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Original data (first two features)
    for label in np.unique(y):
        axes[0].scatter(
            X_original[y == label, 0],
            X_original[y == label, 1],
            label=f'Class {label}',
            alpha=0.7
        )
    axes[0].set_xlabel('Sepal length (cm)')
    axes[0].set_ylabel('Sepal width (cm)')
    axes[0].set_title('Original Data (first two features)')
    axes[0].legend()

    # PCA-projected data
    for label in np.unique(y):
        axes[1].scatter(
            X_pca[y == label, 0],
            X_pca[y == label, 1],
            label=f'Class {label}',
            alpha=0.7
        )
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].set_title('PCA Projection')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

@app.command()
def main(
    n_components: int = typer.Option(2, "--n-components", "-n", help="Number of components to keep"),
):
    # 1. Standardize the data
    X = iris.data # type: ignore
    y = iris.target # type: ignore
    X_centered = (X - np.mean(X, axis = 0)) / np.std(X, axis = 0)
    
    # 2. Compute the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)
    
    # 3. Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # 4. Project the data onto the principal components
    W = sorted_eigenvectors[:, :n_components]
    X_pca = X_centered @ W
    
    plot_pca(X_pca, y, X_centered)

if __name__ == "__main__":
    app()