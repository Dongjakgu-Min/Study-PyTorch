import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [6], [9]])

# 입력, 출력 차원을 각각 1로 설정
input_dim, output_dim = 1, 1

# PyTorch의 신경망 모델을 구축하는 데 사용되는 Layer 중 하나, 이것은 선형 변환을 수행
# 선형 변환은 입력 데이터에 가중치 행렬과 편향 벡터를 더한 뒤 편향을 더하는 연산임
model = nn.Linear(input_dim, output_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    predictions = model(x_train)

    # pytorch에서 제공하는 평균 제곱 오차 함수
    cost = F.mse_loss(predictions, y_train)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('Epoch: {}, Cost: {}'.format(epoch, cost.item()))
