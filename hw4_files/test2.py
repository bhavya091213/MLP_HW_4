from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import mnist
import torch
import numpy as np
import torchvision

 
def test(dataloader,model):

    #please implement your test code#
    ##HERE##
    ###########################                                                                                                                                                                               

    test_accuracy=0

    print("test accuracy:", test_accuracy)

 

def main():

    pad=torchvision.transforms.Pad(2,fill=0,padding_mode='constant')

    mnist_test=mnist.MNIST(split="test",transform=pad)

    test_dataloader= DataLoader(mnist_test,batch_size=1,shuffle=False)

    model = torch.load("LeNet2.pth")

    test(test_dataloader,model)

 

if __name__=="__main__":

    main()
