import torch, os
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

###############################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].

transform = transforms.Compose([transforms.ToTensor(),])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

###############################################################################
# # ---------------------------------------------------------------------------
# # functions to show an image
# import matplotlib.pyplot as plt
# import numpy as np
#
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# # ---------------------------------------------------------------------------
###############################################################################

# For training over different parameters pushed through state_params to Net()
#-----------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

state_params = {'num_epochs'=15,'learning_rate'= 0.001,'conv_layers'=3}
class ModuleList(list):
    modelslist = []
    def __init__(self,states):
        
