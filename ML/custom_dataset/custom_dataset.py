import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor


torch.manual_seed(0)


class CustomDataset(Dataset):
    # Dataset의 전처리를 해주는 부분
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    # 총 데이터의 개수를 반환
    def __len__(self):
        return len(self.x_data)

    # index를 입력받아 그에 mappping되는 입출력 data를 PyTorch의 Tensor 형태로 반환
    def __getitem__(self, idx):
        return FloatTensor(self.x_data[idx]), FloatTensor(self.y_data[idx])


dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
model = torch.nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {}, Batch {}, Loss {}'.format(epoch, batch_idx, cost.item()))
