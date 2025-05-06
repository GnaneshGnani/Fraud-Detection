import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

class FraudDetectionModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer_1 = torch.nn.Linear(input_size, hidden_size)
        self.relu_1 = torch.nn.ReLU()
        self.layer_2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.relu_1(self.layer_1(x))
        x = self.sigmoid(self.layer_2(x))

        return x

class PyTorchModel(BaseEstimator):
    def __init__(self, input_size, output_size, hidden_size = 32, lr = 0.001, num_epochs = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.num_epochs = num_epochs

        self.model = FraudDetectionModel(input_size, hidden_size, output_size)
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

    def fit(self, X, y):
        self.model.train()
        features = torch.tensor(X, dtype = torch.float)
        labels = torch.tensor(y, dtype = torch.float)

        for epoch in range(self.num_epochs):
            self.optimizer.zero_grad()

            probs = self.model(features).squeeze()
            loss = self.criterion(probs, labels)

            loss.backward()
            self.optimizer.step()
        
        return self

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype = torch.float32)
            probs = self.model(X_tensor).squeeze()
            predicted = (probs > 0.5).float()

        return predicted.numpy()
    
    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype = torch.float32)
            probs = self.model(X_tensor).squeeze()

        return probs.numpy()

    def score(self, X, y):
        y_pred = self.predict(X)

        return accuracy_score(y, y_pred)