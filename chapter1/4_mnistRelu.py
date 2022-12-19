import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

# 1. Device Check
Device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device : {Device}")

# 하이퍼 파라미터
Batch_size = 32
Epochs = 30

# 2. Data download
train_dataset = datasets.MNIST(root="./data",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True
                               )
test_dataset = datasets.MNIST(root="./data",
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True
                               )
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = Batch_size,
                                           shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = Batch_size,
                                           shuffle = False)


# 3. DataCheck
for (x_train, y_train) in train_loader:
    print(f"x_train : {x_train.shape}\t dtype : {type(x_train)}")
    print(f"y_train : {y_train.shape}\t dtype : {type(y_train)}")
    
    sub_plot=1
    plt.figure(figsize=(10*sub_plot, 2*sub_plot))
    for i in range(1,11):
        plt.subplot(1, 10, i)
        plt.axis('off')
        plt.imshow(x_train[i,:,:,:].numpy().reshape(28,28), cmap="gray_r")
        plt.title(f"Class : {y_train[i].item()}")
    plt.show()
    break

# 4. Set the MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = 0.5
        
    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        x = F.dropout(x, training = self.training ,p=self.dropout)
        x = F.relu(x)           # Relu 함수 적용
        x = self.fc2(x)
        x = F.dropout(x, training = self.training ,p=self.dropout)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
    
# 5. 목적함수, 손실함수 정의
model = MLP().to(Device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.5)
criterion = nn.CrossEntropyLoss()

# 6. 모델 훈련 함수 정의
def train(model, train_loader, optimizer, log_interval, Epoch):
    model.train()
    for idx, (x_train, y_train) in enumerate(train_loader):
        data = x_train.to(Device)
        target = y_train.to(Device)
        
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)
        loss.backward()
        optimizer.step()
        
        if idx % log_interval ==0:
            print(f"Train Epoch : {Epoch}\t [{idx*len(data)} / {len(train_loader.dataset)} ({100.*idx/len(train_loader):.6f}%)]\tTrain Loss : {loss.item():.6f}")
            
# 7. 모델 평가 함수 정의
def eval(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(Device)
            target = target.to(Device)
            
            output = model(data)
            test_loss += criterion(output, target).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# 8.MLP 학습 및 검증 수행
for Epoch in range(0, Epochs):
    train(model, train_loader, optimizer,200, Epoch)
    test_loss, test_accuracy = eval(model, test_loader)
    print(f"\nEpochs : {Epoch}\tTest_Loss : {test_loss:.6f}\tTest_accuracy : {test_accuracy}")
            
            
    
    
        