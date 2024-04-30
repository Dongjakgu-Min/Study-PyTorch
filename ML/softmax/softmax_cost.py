import torch
import torch.nn.functional as F

torch.manual_seed(1)

#  Implement Softmax - Low Level
z = torch.FloatTensor([1, 2, 3])
hypothesis = F.softmax(z, dim=0)
print(hypothesis)
print("hypothesis sum: ", hypothesis.sum())  # 합은 1

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)  # 각 행의 원소들의 합은 1

#  각 샘플에 대해서 임의의 레이블 생성
y = torch.randint(5, (3, )).long()  # 0에서 4 사이의 요소를 세 개 갖는 64Bit 정수 텐서를 생성
print(y)

y_one_hot = torch.zeros_like(hypothesis)  # 모든 원소가 0의 값을 가진 3*5 텐서를 만듦, hypothesis와 같은 모양의 텐서를 만듦
y_one_hot.scatter_(1, y.unsqueeze(1), 1)  # 첫번째 인자는 차원, 두번째 인자는 인덱스, 세번째 인자는 할당할 값.

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

# Implement Softmax - High Level
F.log_softmax(z, dim=1)  # Softmax 함수의 출력값을 로그 함수의 입력으로 사용
(y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()  # Cost 함수
F.nll_loss(F.log_softmax(z, dim=1), y)  # nll = Negative Log Likelihood
F.cross_entropy(z, y)  # F.cross_entropy는 비용 함수에 Softmax 함수까지 포함하고 있음을 기억하고 있어야 구현 시 혼동하지 않음.
