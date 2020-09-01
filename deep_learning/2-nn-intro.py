import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from deep_learning import helper
from deep_learning.basic_nn import Network

def activation(x):
    return 1/(1+torch.exp(-x))

def softmax(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
dataiter = iter(trainloader)
# total = sum(1 for _ in dataiter) and total = 938
images, labels = dataiter.next()    # first batch only
print('image type = {}'.format(type(images)))
print('images.shape = {}'.format(images.shape))
print('labels.shape = {}'.format(labels.shape))
#plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

images.resize_(64, 1, 784)
img_idx = 0

model = Network()
ps = model.forward(images[img_idx, :])
print('probabilities.shape = {}'.format(ps))
img = images[img_idx]
helper.view_classify(img.view(1, 28, 28), ps)
