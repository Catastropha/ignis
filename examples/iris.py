import torch.nn as nn
import torch.optim as optim
import pandas as pd

from ignis import fit
from ignis.callbacks import EarlyStop, ModelCheckpoint


df = pd.read_csv('examples/iris.csv')
data = df.drop(columns=['Id', 'Species'])
labels = df['Species']
labels = pd.get_dummies(labels)


class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


model = Model()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

fit(
    x=data,
    y=labels,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epoch=500,
    batch_size=8,
    verbose=True,
)
