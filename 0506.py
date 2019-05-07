#%%
import torch
import numpy as np
import cv2
import tensorflow as tf

#%%
print('torch', torch.__version__)
print('tensorflow', tf.__version__)
print('opencv', cv2.__version__)

#%%
# import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as trans

trainSet = dsets.MNIST(root='./data', train=True,
transform=trans.ToTensor(), download=True)

#%%
print('Number of images {}'.format(len(trainSet)))
print('Type {}'.format(type(trainSet[0][0])))
print('Size of each image {}'.format(trainSet[0][0].size()))
print('label {}'.format(trainSet[0][1]))

#%%
class MultiLogisticModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MultiLogisticModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

in_dim = 28*28
out_dim = 10
model = MultiLogisticModel(in_dim, out_dim)
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=.001)

#%%
batchSize = 100
epochs = 5
trainloader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batchSize, shuffle=True)

#%%
for epoch in range(epochs):

    runningLoss = 0.0
    # one batch
    for i, (images, labels) in enumerate(trainloader):
        images = images.view(-1, 28*28)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()
        runningLoss += loss.item()
    
    print(runningLoss)

#%%
import numpy as np
def successRate(predicted, labes):
    predict = [np.argmax(p.detach().numpy()) for p in predicted]
    actual = [labels[i].item() for i in range(len(predicted))]
    correct = [i for i, j in zip(predict, actual) if i==j]
    return (len(correct)/(len(predict)))

#%%
testSet = dsets.MNIST(root='./data', train=False, transform=trans.ToTensor(), download=True)
testloader = torch.utils.data.DataLoader(dataset=testSet, batch_size=10000, shuffle=True)

#%%
testData = iter(testloader)
images, labels = testData.next()
print(images.shape)
print(labels.shape)
print(type(labels[0].item()))
print(labels[0].item())
output = model(images.view(-1, 28*28))
print(successRate(output, labels))

#%%
