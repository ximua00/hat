import sys
sys.path.append('..')
import torch

import utils

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla):
        super(Net,self).__init__()

        _,n_channels,size,_=inputsize
        self.taskcla=taskcla

        self.conv1=torch.nn.Conv2d(n_channels,64,kernel_size=size//8)
        s=utils.compute_conv_output_size(size,size//8)
        s=s//2
        self.conv2=torch.nn.Conv2d(64,128,kernel_size=size//10)
        s=utils.compute_conv_output_size(s,size//10)
        s=s//2
        self.conv3=torch.nn.Conv2d(128,256,kernel_size=2)
        s=utils.compute_conv_output_size(s,2)
        s=s//2
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)
        self.fc1=torch.nn.Linear(256*s*s,2048)
        self.fc2=torch.nn.Linear(2048,2048)
        self.last=torch.nn.ModuleList()
        for t,output_classes in self.taskcla:
            self.last.append(torch.nn.Linear(2048,output_classes))

        return

    def forward(self,x):
        h=self.maxpool(self.drop1(self.relu(self.conv1(x))))
        h=self.maxpool(self.drop1(self.relu(self.conv2(h))))
        h=self.maxpool(self.drop2(self.relu(self.conv3(h))))
        h=h.view(x.size(0),-1)
        h=self.drop2(self.relu(self.fc1(h)))
        h=self.drop2(self.relu(self.fc2(h)))
        y=[]
        for t,i in self.taskcla:
            y.append(self.last[t](h))
        return y




if __name__ == "__main__":
    size = (1,1,28,28)
    input = torch.rand(size)

    net = Net(inputsize=size, taskcla=[(0, 5), (1, 5)])
    print(net)
    
    output = net(input)
    print(output[0].size())