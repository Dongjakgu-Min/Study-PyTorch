import torch
import torch.nn as nn


x = torch.ones(5)  # 입력 Tensor
y = torch.zeros(3)  # 정답 Tensor
w = torch.randn(5, 3, requires_grad=True)  # 세로 5, 가로 3의 무작위 값의 행렬을 만든다
b = torch.randn(3, requires_grad=True)  # 세로 1, 가로 3의 무작위 행렬을 만든다
z = torch.matmul(x, w) + b  # 행렬곱

z_det = z.detach()  # 

loss_fn = nn.MSELoss()  # 손실 함수

loss = loss_fn(z, y)  # 손실 함수와 행렬곱을 비교하여 평균제곱 오차를 구함

loss.backward()  # 평균 제곱 오차를 이용하여 역전파 알고리즘 사용

print(w.grad)
print(b.grad)
print(z_det.requires_grad)
