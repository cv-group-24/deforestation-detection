from sklearn.tree import DecisionTreeClassifier
import torch.nn as nn
import torch

class MultiModalMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MultiModalMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images, x):
        x = self.sigmoid(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DecisionTreeWrapper(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DecisionTreeWrapper, self).__init__()
        self.model = DecisionTreeClassifier(max_depth=5)

    def forward(self, images, x):
        x_np = x.detach().cpu().numpy()  # Convert tensor to NumPy
        preds = self.model.predict(x_np)
        return torch.tensor(preds, dtype=torch.float32).to(x.device)

    def fit(self, x, y):
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        self.model.fit(x_np, y_np)

    def predict(self, x):
        x_np = x.detach().cpu().numpy()
        return torch.tensor(self.model.predict(x_np), dtype=torch.float32).to(x.device)
    

from sklearn.ensemble import RandomForestClassifier

class RandomForestWrapper(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RandomForestWrapper, self).__init__()
        self.model = RandomForestClassifier(n_estimators=100)

    def forward(self, images, x):
        x_np = x.detach().cpu().numpy()
        preds = self.model.predict(x_np)
        return torch.tensor(preds, dtype=torch.float32).to(x.device)

    def fit(self, x, y):
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        self.model.fit(x_np, y_np)

    def predict(self, x):
        x_np = x.detach().cpu().numpy()
        return torch.tensor(self.model.predict(x_np), dtype=torch.float32).to(x.device)