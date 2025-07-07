import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from torch.utils.data import TensorDataset, DataLoader
import typer

app = typer.Typer()

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X, y)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
def accuracy(y_pred, y_true):
    y_pred_np = y_pred.detach().numpy().flatten()
    y_true_np = y_true.detach().numpy().flatten()
    return np.round(np.mean(y_pred_np == y_true_np), 2)

def predict_label(y_pred):
    return (y_pred >= 0.5)

@app.command()
def train(
    epochs: int = typer.Option(100, "--epochs", "-e", help="Number of epochs"),
    learning_rate: float = typer.Option(0.01, "--learning-rate", "-l", help="Learning rate"),
):
    model = XORModel()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for X, y in dataloader:
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    y_pred = predict_label(model(X))
                    acc = accuracy(y_pred, y)
                    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc}")
                model.train()
    
    print(model)

if __name__ == "__main__":
    app()