import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

class PeekLayer(nn.Module):
    def __init__(self):
        super(PeekLayer, self).__init__()
        self.lin = nn.Linear(2,2)
        #
    def forward(self, x):
        return self.lin(x)

X = torch.randn(400,2)
y = X**2

# train-test split of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)#.reshape(-1, 1)
ds_train = torch.utils.data.TensorDataset(X_train, y_train)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)#.reshape(-1, 1)
ds_test = torch.utils.data.TensorDataset(X_train, y_train)

dl_train = DataLoader(ds_train, batch_size=64, shuffle=True)
dl_test = DataLoader(ds_test, batch_size=64, shuffle=True)

# Create model
model = PeekLayer()

# Loss and Optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training parameters
n_epochs = 10000

# Hold the best model
best_testloss_total = np.inf   # init to infinity
best_weights = None
history = []

# training loop
for epoch in range(n_epochs):

    loss_total = 0
    model.train()
    for X_batch, y_batch in dl_train:

        # forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # update weights
        optimizer.step()

        loss_total += float(loss)

    # evaluate accuracy at end of each epoch
    model.eval()
    testloss_total = 0
    for X_test, y_test in dl_train:
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        testloss_total += float(mse)
    history.append(testloss_total)
    if testloss_total < best_testloss_total:
        best_testloss_total = testloss_total
        best_weights = copy.deepcopy(model.state_dict())

    if epoch % 10 == 0:
        print(epoch, "\t", loss_total, "\t", testloss_total)
# restore model and return best accuracy
model.load_state_dict(best_weights)