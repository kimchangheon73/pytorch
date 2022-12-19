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

# 6. 파이토치 내에서 기본 제공하는 모델을 불러와 FC충 추가 및 output크기 설정
import torchvision.models as models
model = models.resnet34(pretrained=False)             # 모델 구조만 불러옴 (파라미터는 랜덤)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.cuda()
    
# 7. Optimizer, Loss
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