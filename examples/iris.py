import torch.nn as nn
import torch.optim as optim
import pandas as pd

from ignis import fit
from ignis.loaders import create_loaders
from ignis.callbacks import EarlyStop, ModelCheckpoint


df = pd.read_csv('examples/iris.csv')
data = df.drop(columns=['Id', 'Species'])
labels = df['Species']
labels = pd.get_dummies(labels)

train_loader, validation_loader = create_loaders(
    x=data.values,
    y=labels.values,
    validation_split=0.1,
)


class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


model = Model()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
callbacks = [
    EarlyStop(monitor='train_loss', patience=3),
    ModelCheckpoint(monitor='validation_loss', filepath='best_model.pt'),
]

fit(
    train_loader=train_loader,
    validation_loader=validation_loader,
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=500,
    callbacks=callbacks,
    verbose=True,
)
