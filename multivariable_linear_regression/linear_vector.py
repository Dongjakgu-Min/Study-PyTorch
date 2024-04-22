import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)  # 랜덤 시드 고정

# H(x) = w1x1 + w2x2 + w3x3 + b
# 3차원 데이터 x1 x2 x3
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 80],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor(([152], [185], [180], [196], [142]))
# 가중치 w
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)  # 편향

optimizer = optim.SGD([W, b], lr=1e-5)  # optimizer 설정
nb_epochs = 10000  # 반복 횟수

for epoch in range(nb_epochs + 1):
    hypothesis = x_train.matmul(W) + b  # 가설

    cost = torch.mean((hypothesis - y_train) ** 2)  # cost 계산 (가설과 학습용 y 차이의 평균

    optimizer.zero_grad()  # optimizer gradient 초기화
    cost.backward()  # 역전파 알고리즘 사용
    optimizer.step()  # 역전파 단계에서 수집된 변화도로 매개변수를 조정

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))
