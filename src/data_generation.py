import os
from torchvision import datasets
import torchvision
import torch
from torchvision.transforms import transforms
import torchvision.models as models


def datagen():

    batch_size = 64
    num_workers = 0
    transform = transforms.Compose([

        transforms.Resize(size=(256,256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    ])


    train_set = '/home/surya/F/Fire_and_Smoke/train'
    valid_set = '/home/surya/F/Fire_and_Smoke/test'


    train_data = datasets.ImageFolder(train_set, transform=transform)
    valid_data = datasets.ImageFolder(valid_set, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }


    
    return loaders
   
if __name__=='__main__':
    datagen()

