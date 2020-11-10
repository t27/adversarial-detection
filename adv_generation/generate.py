""" CIFAR10 Adversarial Example Generation"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import numpy as np
import wandb

from tqdm import tqdm
from utils import parse_args

import _init_paths
from detector import base_model

DATA_DIR = "../detector/data"
ADV_DIR = "./baseline_adv/"
MODEL_PATH = "../detector/savedmodels/epoch_56.pth"
SIZE = 32

idx_to_class_map = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}


def basemodel_main():
    """Main Function for BaseModel"""
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # load model
    model = nn.DataParallel(base_model.BaseModel(num_classes=10))
    model.load_state_dict(torch.load(MODEL_PATH))

    # for eps in epsilons:
    #     generate(eps, model)
    generate(0.15, model, "train")
    generate(0.15, model, "dev")


def resnet_main():
    """Main function."""
    # magnitude of chan
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # load model
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load("saved_models/res80.pth"))

    # for eps in epsilons:
    #     generate(eps, model)
    generate(0.15, model)


def generate(epsilon=0.0, model=None, phase="dev", args=None):
    """Run one training session with a particular set of hyperparameters."""

    args, device = parse_args()

    criterion = nn.CrossEntropyLoss()

    dataloaders, idx_to_class = get_cifar_loader()

    # dataloaders, idx_to_class = get_loader(args)

    model.to(device)
    model.eval()

    running_corrects = 0.0
    adv_corrects = 0.0
    counter = np.zeros(10)

    # Iterate over data.
    for inputs, labels in tqdm(dataloaders[phase]):

        inputs = inputs.to(device)
        labels = labels.to(device)

        inputs.requires_grad = True

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()

        data_grad = inputs.grad.data

        perturbed_data = fgsm_attack(inputs, epsilon, data_grad)

        adv_outputs = model(perturbed_data)
        _, adv_preds = torch.max(adv_outputs, 1)

        save_images(
            perturbed_data, labels, preds, adv_preds, idx_to_class, counter, phase
        )

        # statistics
        rc = torch.sum(preds == labels.data)
        running_corrects += rc.item() / preds.shape[0]

        # adversarial statistics
        arc = torch.sum(adv_preds == labels.data)
        adv_corrects += arc.item() / preds.shape[0]

    original_acc = running_corrects / len(dataloaders[phase])
    adversarial_acc = adv_corrects / len(dataloaders[phase])

    print(
        "Original Accuracy: {:.4f} \t Adversarial Accuracy: {:.4f}".format(
            original_acc, adversarial_acc
        )
    )


def save_images(images, labels, orig_pred, adv_pred, idx_to_class, counter, phase):
    """Take in tensor of images"""
    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.247, 1 / 0.243, 1 / 0.261]
            ),
            transforms.Normalize(mean=[-0.4914, -0.4822, -0.4465], std=[1.0, 1.0, 1.0]),
        ]
    )

    for i in range(images.shape[0]):
        # number corresponds to original label
        counter[labels[i].item()] += 1
        # saved only intentionally misclassified examples
        if orig_pred[i] == labels[i] and adv_pred[i] != labels[i]:
            image = invTrans(images[i])
            image = image.cpu().detach().numpy()
            image = np.transpose(image, (1, 2, 0))
            image = np.clip(image, 0, 1)

            name = str(int(counter[labels[i].item()]))
            name = name.zfill(4) + ".png"
            save_dir = os.path.join(ADV_DIR, phase, idx_to_class[labels[i].item()])
            if not os.path.exists((save_dir)):
                os.makedirs(save_dir)
            save_file = os.path.join(save_dir, name)
            plt.imshow(image)
            plt.imsave(save_file, image)
        else:
            continue


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    low = torch.min(image).item()
    high = torch.max(image).item()
    perturbed_image = torch.clamp(perturbed_image, low, high)
    # Return the perturbed image
    return perturbed_image


def get_cifar_loader(BATCH_SIZE=32):
    """Load the CIFAR dataset via torchvision"""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6
    )

    dev_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6
    )
    return {"train": train_loader, "dev": dev_loader}, idx_to_class_map


def get_loader(args):
    """Return Image Loaders as a Dictionary."""
    size = SIZE

    resize = transforms.Resize(size=int(SIZE * 1.25), interpolation=2)
    crop = transforms.RandomResizedCrop(
        size=SIZE, scale=(0.7, 1.0), ratio=(0.75, 1.3), interpolation=2
    )
    gray = transforms.RandomGrayscale(p=0.1)
    flip = transforms.RandomHorizontalFlip(p=0.5)
    jitter = transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
    )
    rot = transforms.RandomRotation(
        degrees=15, resample=False, expand=False, center=None, fill=None
    )
    affine = transforms.RandomAffine(degrees=15)

    # transform = transforms.Compose([resize, crop, gray, flip, jitter, rot, affine, transforms.ToTensor()])

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
        ]
    )

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), tf)
        for x in ["train", "test"]
    }
    if args.fast:
        image_datasets["train"] = image_datasets["test"]

    class_to_idx = image_datasets["train"].class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=args.batch_size, shuffle=False, num_workers=4
        )
        for x in ["train", "test"]
    }

    return dataloaders, idx_to_class


if __name__ == "__main__":
    basemodel_main()
