# 1. Module Import 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

# 2. check the device 
Device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device : {Device}")

# 3. hyper parameter
Batch_size = 32
epochs = 10

# 4. Loading data
train_dataset = datasets.CIFAR10(root="./data",
                                 download=True,
                                 train=True,
                                 transform=transforms.Compose([                 # compose : 이미지 전처리, agumentation 적용 시 사용하는 매서드
                                     transforms.RandomHorizontalFlip(),         # 이미지를 50% 확률로 좌우 반전
                                     transforms.ToTensor(),                     # Tensor 형태로 변환 
                                     transforms.Normalize((0.5,0.5,0.5),         # 텐서 이미지에 다른 정규화를 진행 r g b 평균을 0.5씩 적용  
                                                          (0.5,0.5,0.5))]))     # r g b 이미지 표준편차를 0.5씩 적용
test_dataset = datasets.CIFAR10(root="./data",
                                 download=True,
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
train_loader = torch.utils.data.DataLoader(train_dataset,
                                      batch_size = Batch_size,
                                      shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                      batch_size = Batch_size,
                                      shuffle = False)

# 5. check the data 
for (x_train, y_train) in train_loader:
    print(f"train Data shape : {x_train.shape}\tData dtype : {type(x_train)}")
    print(f"target Data shape : {y_train.shape}\tData dtype : {type(y_train)}")
    
    pltsize = 1
    plt.figure(figsize=(10*pltsize, pltsize*3))
    
    for i in range(10):
        plt.subplot(1,10,i+1)
        plt.axis("off")
        plt.imshow(np.transpose(x_train[i], (1,2,0)))
        plt.title(f"{y_train[i]}")
    # plt.show()
    break

# 6. ResNet 모델 설계
class BasicBlock(nn.Module):
  def __init__(self, in_planes, planes, stride = 1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_planes, planes,
                           kernel_size = 3,
                           stride = stride,
                           padding = 1,
                           bias = False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes,
                           kernel_size = 3,
                           stride = 1,
                           padding = 1,
                           bias = False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, planes,
                    kernel_size = 1,
                    stride = stride,
                    bias = False),

          nn.BatchNorm2d(planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    
    return out
    
class ResNet(nn.Module):
  def __init__(self, num_classes = 10):
    super(ResNet, self).__init__()
    self.in_planes = 16

    self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(16)
    self.layer1 = self._make_layer(16, 2, stride=1)
    self.layer2 = self._make_layer(32, 2, stride = 2)
    self.layer3 = self._make_layer(64, 2, stride = 2)
    self.linear = nn.Linear(64, num_classes)


  def _make_layer(self, planes, num_blocks, stride):
    stride = [stride] + [1] * (num_blocks - 1)
    layers = []

    for stride in stride:
      layers.append(BasicBlock(self.in_planes, planes, stride))
      self.in_planes = planes
    return nn.Sequential(*layers)

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1)
    out = self.linear(out)

    return out
    
    
    
    
    
# 7. Optimizer, Loss
model = ResNet().to(Device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss()
print(model)

# 8. Train
def train(model, train_loader, optimizer, log_interval):
    model.train()
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        image = x_train.to(Device)
        label = y_train.to(Device)
        optimizer.zero_grad()
        
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print(f"Train epochs : {epochs}\t[{batch_idx*len(image):<5}/{len(train_loader.dataset)}({100.*batch_idx/len(train_loader):.2f}%)]\tLoss : {loss.item()}")
        
# 9. Eval
def eval(model, test_loader, optimizer, log_interval):
    model.eval()
    test_loss = 0
    correct = 0 
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(Device)
            y_test = y_test.to(Device)
            output = model(x_test)
            
            test_loss += criterion(output, y_test).item()
            prediction = output.max(1, keepdim = True)[1]
            correct += prediction.eq(y_test.view_as(prediction)).sum().item()
            
    test_loss  /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy 
            
# 10. Start
for Epoch in range(1, epochs+1):
    train(model, train_loader, optimizer, 200)
    test_loss, test_accuracy = eval(model, test_loader, optimizer, 200)
    print(f"\nEpoch : {Epoch}\tTest_loss : {test_loss}\tTest_accuracy : {test_accuracy}%\n")