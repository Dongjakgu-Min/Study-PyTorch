import torch
import torch.nn as nn
import torch.nn.functional as F


x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#  선형 모델 선언 및 초기화. x와 y 모두 1차원이므로 1 설정
model = nn.Linear(1, 1)
#  최적화 방법으로는 경사하강법 사용
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # lr = 학습률(learning rate)

nb_epochs = 4000
for epoch in range(nb_epochs + 1):
    prediction = model(x_train)  # H(x) 계산
    cost = F.mse_loss(prediction, y_train)  # cost 계산

    #  cost로 H(x) 개선하는 부분
    optimizer.zero_grad()  # gradient 초기화

    #  비용 함수를 미분하여 gradient 계산
    cost.backward()  # backward 계산

    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch: {}, Cost: {:.6f}'.format(epoch, cost.item()))

new_var = torch.FloatTensor([[4.0]])
prediction = model(new_var)
print(prediction)

