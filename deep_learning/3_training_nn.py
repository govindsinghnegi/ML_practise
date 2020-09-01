import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import optim

from deep_learning import helper

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))
# Define the loss
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # TODO: Training pass
        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print('Training loss: {}'.format(running_loss/len(trainloader)))



images, labels = next(iter(trainloader))
img = images[0].view(1, 784)
# Turn off gradients to speed up this part
with torch.no_grad():
    logps = model.forward(img)

# Output of the network are logits, need to take softmax for probabilities
ps = torch.exp(logps)
helper.view_classify(img.view(1, 28, 28), ps)


##### gradient demo #####
print('\n ############# gradient demo ############## \n')
x = torch.randn(2, 2, requires_grad=True)
print('x = {}'.format(x))
y = x**2
print('y = {}'.format(y))
print('y.grad_fn = {}'.format(y.grad_fn))
z = y.mean()
print('z = {}'.format(z))
print('x.grad = {}'.format(x.grad))
# trigger back propagation
z.backward()
print('after back propagation')
print('z.grad = {}'.format(z.grad))
print('x.grad = {}'.format(x.grad))
print('x/2 = {}'.format(x/2))

###############################