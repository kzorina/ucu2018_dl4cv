from src.models.model_factory import ModelFactory
import yaml
from PIL import Image
import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch
import torch.nn as nn
from src.utils import handle_device

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
fin_scale=32
init_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Resize(fin_scale),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean,
                             std=imagenet_std)
    ])
with open('C:\\Users\\kzorina\\Studing\\CV\\6DLinCV\\ucu2018_dl4cv\\assignments\\classification\\experiments\\cfgs\\params_fashion.yaml', 'r') as f:
    params = yaml.load(f)
    print(params)
params['device'] = handle_device(params['with_cuda'])
device = params['device']
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = init_transform(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)  #assumes that you're using GPU

image = image_loader("C:\\Users\\kzorina\\Studing\\CV\\6DLinCV\\ucu2018_dl4cv\\assignments\\classification\\data\\test.jpg")
model = ModelFactory.create(params)
model.load_state_dict(torch.load('C:\\Users\\kzorina\\Studing\\CV\\6DLinCV\\ucu2018_dl4cv\\assignments\\classification\\experiments\\e0008_model_best.pth'))
result = model(image).data.numpy()
print(result)
#import numpy as np
#array_res = np.reshape(result, (32,32))
'''

criterion = nn.CrossEntropyLoss().to(params['device'])
data = dsets.ImageFolder(root='C:\\Users\\kzorina\\Studing\\CV\\6DLinCV\\ucu2018_dl4cv\\assignments\\classification\\data\\deepfashion\\custom\\',
    transform=init_transform,
    target_transform=None)
model.eval()

with torch.no_grad():
    for i, (input_data, target) in enumerate(data):
        target = target.to(device)
        input_data = input_data.to(device)

        # compute output
        output = model(input_data)
        loss = criterion(output, target)
        print(output)
        # measure accuracy and record loss




def __init__(self,
             path_data,
             num_dunkeys=4,
             batch_size_train=100,
             batch_size_val=100,
             fin_scale=32):


    

    self.transforms = {
        'train': init_transform,
        'val': init_transform
    }

    self.dataset = {
        'train': dsets.ImageFolder(root=os.path.join(path_data, 'custom'),
                                   transform=self.transforms['val'],
                                   target_transform=None),
        'val': dsets.ImageFolder(root=os.path.join(path_data, 'test'),
                                 transform=self.transforms['val'],
                                 target_transform=None)
    }

    self.loader = {
        'train': torch.utils.data.DataLoader(dataset=self.dataset['train'],
                                             batch_size=batch_size_train,
                                             shuffle=True,
                                             num_workers=num_dunkeys),
        'val': torch.utils.data.DataLoader(dataset=self.dataset['val'],
                                           batch_size=batch_size_val,
                                           shuffle=False,
                                           num_workers=num_dunkeys)
    }
output = model(input_data)

'''