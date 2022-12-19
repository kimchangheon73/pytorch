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
                                 transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root="./data",
                                 download=True,
                                 train=False,
                                 transform=transforms.ToTensor())
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
    break

# 6. CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.fc1 = nn.Linear(8*8*16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        
                                # size    : cnt x width x height x depth
    def forward(self, x):       # x_shape : 32 x (32 x 32 x 3) 
        x = self.conv1(x)       # conv1 :   32 x (32 x 32 x 8)
        x = F.relu(x)
        x = self.pool(x)        # pooling : 32 x (16 x 16 x 8)
        x = self.conv2(x)       # conv2 :   32 x (16 x 16 x 16)
        x = F.relu(x)
        x = self.pool(x)        # pooling : 32 x (8 x 8 x 16)
        x = x.view(-1, 8*8*16)  # view : 32 x (-1 x 8*8*16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        
        return x
    
# 7. Optimizer, Loss
model = CNN().to(Device)
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