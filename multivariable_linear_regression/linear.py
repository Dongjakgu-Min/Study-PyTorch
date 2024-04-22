import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(1)  # 랜덤 시드 고정

# H(x) = w1x1 + w2x2 + w3x3 + b
# 3차원 데이터 x1 x2 x3
x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])  # x1
x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])  # x2
x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])  # x3
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])  # y

# 가중치 w
w1 = torch.zeros(1, requires_grad=True)  # 가중치 w1
w2 = torch.zeros(1, requires_grad=True)  # 가중치 w2
w3 = torch.zeros(1, requires_grad=True)  # 가중치 w3
b = torch.zeros(1, requires_grad=True)  # 편향

optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)  # optimizer 설정
nb_epochs = 1000  # 반복 횟수

for epoch in range(nb_epochs + 1):
    hypothesis = x1_train + w1 * x1_train + w2 * x2_train + w3 * x3_train + b  # 가설

    cost = torch.mean((hypothesis - y_train) ** 2)  # cost 계산 (가설과 학습용 y 차이의 평균

    optimizer.zero_grad()  # optimizer gradient 초기화
    cost.backward()  # 역전파 알고리즘 사용
    optimizer.step()  # 역전파 단계에서 수집된 변화도로 매개변수를 조정

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
            epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
        ))
