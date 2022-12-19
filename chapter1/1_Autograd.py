import torch

Device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using DEVICE : {Device}")

# 하이퍼파라미터
Batch_size = 64
Input_size = 1000
Output_szie = 10
Hidden_size = 100

X = torch.randn(Batch_size,
                Input_size,
                device=Device,
                dtype = torch.float,
                requires_grad=False)

w1 = torch.randn(Input_size,
                Hidden_size,
                device=Device,
                dtype = torch.float,
                requires_grad=True)

w2 = torch.randn(Hidden_size,
                Output_szie,
                device=Device,
                dtype = torch.float,
                requires_grad=True)
y = torch.randn(Batch_size,
                Output_szie,
                device=Device,
                dtype = torch.float,
                requires_grad=False)

learning_rate = 1e-6

for t in range(1,501):
    y_pred = X.mm(w1).clamp(min=0).mm(w2)
    
    loss = (y_pred-y).pow(2).sum()
    if t%100 == 0:
        print(f"Iteration : {t}\t Loss : {loss.item()}")
    
    loss.backward()
    
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        
        w1.grad.zero_()
        w2.grad.zero_()
        