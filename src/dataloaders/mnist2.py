import os,sys
import numpy as np
import torch
from torchvision import datasets,transforms
from pprint import pprint as pp

########################################################################################################################

def get(seed=0,fixed_order=False,pc_valid=0):
    data={}
    taskcla=[]
    size=[1,28,28]

    # MNIST
    mean=(0.1307,)
    std=(0.3081,)
    dat={}
    dat['train']=datasets.MNIST('../dat/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    dat['test']=datasets.MNIST('../dat/',train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    data[0]={}
    data[0]['name']='mnist-0-4'
    data[0]['ncla']=5
    data[1]={}
    data[1]['name']='mnist-5-9'
    data[1]['ncla']=5
    for s in ['train','test']:
        loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
        data[0][s]={'x': [],'y': []}
        data[1][s]={'x': [],'y': []}
        for image,target in loader:
            label=target.numpy()[0]
            if label<5:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(label)
            else:
                data[1][s]['x'].append(image)
                data[1][s]['y'].append(label-5)

    # "Unify" and save
    for n in [0,1]:
        for s in ['train','test']:
            data[n][s]['x']=torch.stack(data[n][s]['x']).view(-1,size[0],size[1],size[2])
            data[n][s]['y']=torch.LongTensor(np.array(data[n][s]['y'],dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid']={}
        data[t]['valid']['x']=data[t]['train']['x'].clone()
        data[t]['valid']['y']=data[t]['train']['y'].clone()

    # Others
    n=0
    for t in data.keys():
        taskcla.append((t,data[t]['ncla']))
        n+=data[t]['ncla']
    data['ncla']=n

    return data,taskcla,size

########################################################################################################################


if __name__ == "__main__":
    data,taskcla,size = get()
    # pp(data[0]["train"]["y"].size())
    # pp(data[0]["train"]["x"].size())

    # pp(data[1]["train"]["y"].size())
    # pp(data[1]["train"]["x"].size())
    pp(data[0]["ncla"])
    pp(data[1]["ncla"])
    # Dictionary of datapoints where:
    # data[k1][k2][k3]
    # k1: task id (0, 1)
    # k2: task id (train, test, valid, nclas)
    # k3: task id (x,y)
    print(taskcla)
    print(size)