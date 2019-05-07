#%%
import torch
import torchvision
import torchvision.transforms as transforms

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', download=True, transform = transforms.ToTensor())
trainset

#%%
import matplotlib.pyplot as plt
%matplotlib inline

torchimage = trainset[0][0]
npimage=torchimage.permute(1,2,0)
print(type(npimage))
plt.imshow(npimage, interpolation="none")

#%%
import csv
from torch.utils.data import Dataset, DataLoader
class toyDataset(Dataset):
    def __init__(self, dataPath, labelsFile, transform=None):
        self.dataPath = dataPath
        self.transform = transform
        self.labels = []

        with open(os.path.join(self.dataPath, labelsFile)) as f:
            for i, line in enumerate(csv.reader(f)):
                label = tuple(line)
                imagefile = dataPath + '/' + label[0]
                print(imagefile)
                if os.path.isfile(imagefile):
                    print('added ',imagefile)
                    self.labels += label

import os
datapath = os.getcwd() + '/github-eks/Deep-Learning-with-PyTorch-Quick-Start-Guide/Chapter01/GuiseppeToys'

toydata = toyDataset(datapath, 'labels.csv', transform=transforms.ToTensor())
toydata.labels
