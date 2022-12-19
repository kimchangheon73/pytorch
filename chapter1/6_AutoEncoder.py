import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Device Check
Device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device : {Device}")

# 하이퍼 파라미터
Batch_size = 32
Epochs = 10

# 2. Data download
train_dataset = datasets.FashionMNIST(root="./data",
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True
                               )
test_dataset = datasets.FashionMNIST(root="./data",
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


# 4. AutoEncoder 모델 설계하기
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,32),)
        
        self.decoder = nn.Sequential(
            nn.Linear(32,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,28*28),
        )
        
    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder
    
model = AE().to(Device)
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
criterion = nn.MSELoss()

# 6. 모델 훈련 함수 정의
def train(model, train_loader, optimizer, log_interval, Epoch):
    model.train()
    for idx, (x_train, _ ) in enumerate(train_loader):
        image = x_train.view(-1,28*28).to(Device)
        target = x_train.view(-1,28*28).to(Device)
        
        optimizer.zero_grad()
        encode, decode = model(image)
        loss = criterion(decode, target)
        loss.backward()
        optimizer.step()
        
        if idx % log_interval ==0:
            print(f"Train Epoch : {Epoch}\t [{idx*len(image)} / {len(train_loader.dataset)} ({100.*idx/len(train_loader):.6f}%)]\tTrain Loss : {loss.item():.6f}")
            
# 7. 모델 평가 함수 정의
def eval(model, test_loader):
    model.eval()
    test_loss = 0
    real_image = []
    gen_image = []
    
    with torch.no_grad():
        for image, _ in test_loader:
            image = image.view(-1,28*28).to(Device)
            target = image.view(-1,28*28).to(Device)
            
            encode, decode  = model(image)
            test_loss += criterion(decode, target).item()
            real_image.append(image.to(Device))
            gen_image.append(decode.to(Device))
                
    test_loss /= len(test_loader.dataset)
    return test_loss, real_image, gen_image


# 8. 학습 및 검증 수행
for Epoch in range(0, Epochs):
    train(model, train_loader, optimizer,200, Epoch)
    test_loss, real_image, gen_image = eval(model, test_loader)
    print(f"\nEpochs : {Epoch}\tTest_Loss : {test_loss:.6f}")
    f, a = plt.subplots(2, 10, figsize=(10,4))
    for i in range(10):
        img = np.reshape(real_image[0][i], (28,28))
        a[0][i].imshow(img, cmap="gray_r")
        a[0][i].set_xticks(())
        a[0][i].set_yticks(()) 
        
    for i in range(10):
        img = np.reshape(gen_image[0][i], (28,28))
        a[1][i].imshow(img, cmap="gray_r")
        a[1][i].set_xticks(())
        a[1][i].set_yticks(()) 
    plt.show()
    
        