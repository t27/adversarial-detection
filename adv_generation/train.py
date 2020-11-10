""" CIFAR10 Classification - Training Code."""
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, models, transforms

import numpy as np
import wandb

from tqdm import tqdm
from utils import parse_args

DATA_DIR = "/home/sanil/deeplearning/adversarial-detection/cifar10"
MODEL_PATH = "saved_models/"
SIZE = 32


def train(args=None):
    """Run one training session with a particular set of hyperparameters."""
    
    args, device = parse_args()

    criterion = nn.CrossEntropyLoss()

    dataloaders = get_loader(args)
    model = get_model()
    model.to(device)
    
    optimizer = get_optimizer(model, "sgd", args)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.2)

    best_loss = 1e5

    #start training
    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1) 

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                rc = torch.sum(preds == labels.data)
                running_corrects += rc.item() / preds.shape[0]
                
            
            if phase=='train':
                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects / len(dataloaders[phase])
            else:
                epoch_loss = running_loss / len(dataloaders[phase])
                epoch_acc = running_corrects /len(dataloaders[phase])

                scheduler.step(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), MODEL_PATH+"best.pth")



def get_model(num_classes=10, use_pretrained=True, feature_extracting=False):
    """Return Model"""
    # model = models.mobilenet_v2(pretrained=use_pretrained)
    model = models.resnet34(pretrained=use_pretrained)

    if feature_extracting:    
        for param in model.parameters():
            param.requires_grad = False

    # model.classifier = nn.Linear(model.last_channel, num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_optimizer(network, optimizer, args):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=args.lr, momentum=0.9, weight_decay=1e-3)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=args.lr)

    return optimizer


def get_loader(args):
    """Return Image Loaders as a Dictionary."""
    
    size = SIZE

    resize  = transforms.Resize(size=int(SIZE*1.25), interpolation=2)
    crop    = transforms.RandomResizedCrop(size=SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.3), interpolation=2)
    gray    = transforms.RandomGrayscale(p=0.1)
    flip    = transforms.RandomHorizontalFlip(p=0.5)
    jitter  = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    rot     = transforms.RandomRotation(degrees=15, resample=False, expand=False, center=None, fill=None)
    affine  = transforms.RandomAffine(degrees=15)

    # transform = transforms.Compose([resize, crop, gray, flip, jitter, rot, affine, transforms.ToTensor()])

    tf = {
        'train': transforms.Compose([
            resize, crop,# gray, flip, jitter, rot, affine,
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ]),
        'test': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ]),
        }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), tf[x]) for x in ['train', 'test']}
    if args.fast:
        image_datasets["train"] = image_datasets["test"]

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}
    
    return dataloaders



if __name__ == "__main__":
	train()