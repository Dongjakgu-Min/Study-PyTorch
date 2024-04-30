import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1)  # 무작위 변수 생성 시 환경 요인을 같게 함
x_train = torch.FloatTensor([[73, 80, 75], [93, 88, 93], [89, 91, 90], [96, 98, 100], [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3, 1)  # 입력값이 3차원, 출력값은 1차원
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)  # 최적화 기법으로 경사하강법 사용

nb_epochs = 2000  # 반복횟수
for epoch in range(nb_epochs + 1):
    prediction = model(x_train)  # H(x) 계산

    cost = F.mse_loss(prediction, y_train)  # cost 계산

    # cost로 H(x)를 개선하는 영역
    optimizer.zero_grad()  # gradient를 0으로 초기화
    cost.backward()  # 비용 함수를 미분하여 gradient 계산
    optimizer.step()  # W와 b를 업데이트

    if epoch % 100 == 0:
        print('Epoch: {}, Cost: {}'.format(epoch, cost.item()))

new_var = torch.FloatTensor([73, 80, 75])
prediction = model(new_var)
print(prediction)