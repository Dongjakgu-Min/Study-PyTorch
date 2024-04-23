import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset


#  학습 데이터들
x_train = torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  90],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

dataset = TensorDataset(x_train, y_train)  # 데이터셋 생성
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        #  H(x) 계산
        prediction = model(x_train)

        #  cost 계산
        cost = F.mse_loss(prediction, y_train)

        #  cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {}, Batch {}, Loss {}'.format(epoch, batch_idx, cost.item()))

new_var = torch.FloatTensor([73, 80, 75])
pred_y = model(new_var)
print("Predicted: ", pred_y)
