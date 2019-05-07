#%%
import torch
import torch.nn as nn

class Model4_1(nn.Module):
    def __init__(self):
        super(Model4_1, self).__init__()
        self.lin1 = nn.Linear(784, 100)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.lin1(x)
        out = self.relu(out)
        out = self.lin2(out)
        return out

class Model4_2(nn.Module):
    def __init__(self):
        super(Model4_2, self).__init__()
        self.lin1 = nn.Linear(784, 100)
        self.tanh = nn.Tanh()
        self.lin2 = nn.Linear(100, 10)

    def forward(self, x):
        out = self.lin1(x)
        out = self.tanh(out)
        out = self.lin2(out)
        return out

class Model4_3(nn.Module):
    def __init__(self):
        super(Model4_3, self).__init__()
        self.lin1 = nn.Linear(784, 100)
        self.sigmoid = nn.Sigmoid()
        self.lin2 = nn.Linear(100, 10)
    
    def forward(self, x):
        out = self.lin1(x)
        out = self.sigmoid(out)
        out = self.lin2(out)
    
model4_1 = Model4_1()
model4_2 = Model4_2()
model4_3 = Model4_3()

#%%
import torch.optim as optim
import time

def benchmark(trainLoader, model, epochs=1, lr=0.01):
    model.__init__()
    start = time.time()
    optimiser = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(trainLoader):
            optimiser.zero_grad()
            outputs = model(images.view(-1, 28*28))
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
    print('Accuracy: {0:.4f}'.format(accuracy(trainLoader, model)))
    print('Training time: {0:.2f}'.format(time.time() - start))

def accuracy(testLoader, model):
    correct, total = 0, 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data
            outputs = model(images.view(-1, 28*28))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return(correct/total)

#%%
import torchvision.datasets as dsets
import torchvision.transforms as trans

trainSet = dsets.MNIST(root='./data', train=True,
transform=trans.ToTensor(), download=True)

batchSize = 100
epochs = 5
trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True)

#%%
print('ReLU activation:')
benchmark(trainLoader, model4_1, epochs=5, lr=0.1)
print('Tanh activation')
benchmark(trainLoader, model4_2, epochs=5, lr=0.1)
print('sigmoid activation')
benchmark(trainLoader, model4_3, epochs=5, lr=0.1)