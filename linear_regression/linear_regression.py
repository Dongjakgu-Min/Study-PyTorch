import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 랜덤값 초기화
torch.manual_seed(0)

# 학습시킬 데이터들
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [6], [9]])

# 1차원의 요소들이 0인 텐서 생성
# requires_grad : 텐서의 변화도(gradient)를 추적할지 말지 설정, 여기서는 사용
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


hypothesis = x_train * W + b

# 최적화 알고리즘과 학습률 설정, 평균제곱오차를 사용하여 손실률 계산
cost = torch.mean((hypothesis - y_train) ** 2)
optimizer = optim.SGD([W, b], lr=0.01)

nb_epochs = 1000  # 경사하강법 반복 횟수(전체 훈련 데이터가 학습에 한 번 사용된 주기)
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()  # optimizer가 관리하는 모든 Parameter의 Gradient를 0으로 초기화시킴
    cost.backward()  # 오류 역전파 알고리즘
    optimizer.step()  # Optimizer를 사용하여 Parameter 업데이트

    if epoch% 100 == 0:
        print('Epoch: {}, Cost: {}'.format(epoch, cost))

test_var = torch.FloatTensor([[4.0]])  # 학습이 잘 되었는지 확인하기 위해 테스트용으로 넣는 값
# 입력한 값 4에 대해서 예측값 y를 계산한 후 pred_y에 저장
with torch.no_grad():
    pred_y = test_var * W + b
    print("Result : {}".format(pred_y))
