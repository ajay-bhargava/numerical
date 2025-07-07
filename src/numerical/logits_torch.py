import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import typer

app = typer.Typer()

def load_data():
    data = load_breast_cancer()
    X, y = data.data, data.target # type:ignore
    return X, y

@app.command()
def logits(
    epochs: int = typer.Option(1000, "--epochs", "-e", help="Number of epochs"),
    lr: float = typer.Option(0.01, "--lr", help="Learning rate"),
):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # Model
    model = nn.Linear(X_train.shape[1], 1)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for i in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            preds = (torch.sigmoid(logits) >= 0.5).float()
            acc = (preds == y_train).float().mean().item()
            print(f"Epoch {i}, Loss: {loss.item():.4f}, Accuracy: {acc:.2f}")

    # Evaluate
    with torch.no_grad():
        logits = model(X_test)
        loss = criterion(logits, y_test)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        acc = (preds == y_test).float().mean().item()
        print(f"Test Loss: {loss.item():.4f}")
        print(f"Test Accuracy: {acc:.2f}")

if __name__ == "__main__":
    app()
