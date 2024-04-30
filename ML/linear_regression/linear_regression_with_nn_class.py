import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


torch.manual_seed(0)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [6], [9]])

model = LinearRegression()
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):
    predictions = model(x_train)
    cost = F.mse_loss(predictions, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, Cost: {cost}')

new_var = torch.FloatTensor([[4.0]])
with torch.no_grad():
    predictions = model(new_var)
    print(f'result : {predictions}')