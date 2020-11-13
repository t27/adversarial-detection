import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb


logging.basicConfig(level=logging.INFO)

USE_WANDB = True
adv_net_level = 1
if USE_WANDB:
    wandb.init(project=f"IDL_proj_adv_net_{adv_net_level}")
    params = wandb.config
else:
    params = {}

params["adv_net_level"] = 1
params["round_name"] = "round0"
params["batch_size"] = 256
params["lr"] = 0.001
params["momentum"] = 0.9
params["epochs"] = 80
params["sched_step_every_n_epochs"] = 2
params["sched_mult_lr_by_gamma_everystep"] = 0.85


class AdvDetector(nn.Module):
    def __init__(self, in_channels, pooling1=False, pooling2=False):
        super(AdvDetector, self).__init__()

        self.pooling1 = pooling1
        self.pooling2 = pooling2

        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=96, kernel_size=3, stride=1, padding=1
        )
        self.layer2 = nn.BatchNorm2d(96)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d(2, 2)  # 1st pooling layer
        self.layer5 = nn.Conv2d(
            in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1
        )
        self.layer6 = nn.BatchNorm2d(192)
        self.layer7 = nn.ReLU()
        self.layer8 = nn.MaxPool2d(2, 2)  # 2nd pooling layer
        self.layer9 = nn.Conv2d(
            in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1
        )
        self.layer10 = nn.BatchNorm2d(192)
        self.layer11 = nn.ReLU()
        self.layer12 = nn.Conv2d(
            in_channels=192, out_channels=2, kernel_size=1, stride=1
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.pooling1 == True:
            x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        if self.pooling2 == True:
            x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)

        x = self.adaptive_avg_pool(x)
        return x


class AdvDataset(Dataset):
    def __init__(self, real_npy: str, adv_npy: str):
        real_x = np.load(real_npy)
        adv_x = np.load(adv_npy)
        real_y = np.zeros(real_x.shape[0])
        adv_y = np.ones(adv_x.shape[0])
        self.X_data = np.vstack([real_x, adv_x])
        self.y_data = np.hstack([real_y, adv_y])
        # print(self.X_data.shape, self.y_data.shape)

    def __getitem__(self, index):
        return (
            torch.tensor(self.X_data[index]),
            torch.tensor(self.y_data[index]),
        )

    def __len__(self):
        return len(self.X_data)


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    print_frequency = 1000
    all_outputs = []
    all_targets = []
    base_steps = epoch * len(train_loader.dataset)
    loss_agg = 0
    # running_loss = 0
    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if torch.cuda.is_available():
            input = input.type(torch.FloatTensor).cuda()
            target = target.type(torch.FloatTensor).cuda()
        optimizer.zero_grad()
        output = model(input)
        output = output.reshape(-1, 2)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        # batch_accuracy = calc_accuracy(output, target)
        pred_ans_idx = torch.argmax(F.softmax(output, dim=1), axis=1)
        all_outputs += list(pred_ans_idx.cpu().numpy())
        all_targets += list(target.cpu().numpy())

        batch_size = input.shape[0]
        loss_agg += loss.item() * batch_size
        running_loss = loss_agg / ((i + 1) * batch_size)
        if USE_WANDB:
            wandb.log(
                {"loss": loss.item(), "running_loss": running_loss, "epoch": epoch,}
            )

        if i % print_frequency == 0:
            logging.info(
                f"Epoch:{epoch} | Step:{i}/{len(train_loader)} | Loss: {loss.item()}"  # " | Batch Acc:{batch_accuracy}"
            )
        # tb_writer.add_scalar("Train/Loss", loss.item(), base_steps + i)
        del input
        del target
        del loss

    accuracy = accuracy_score(all_targets, all_outputs)
    logging.info(
        f"Epoch: {epoch} | Train Accuracy: {accuracy} | Running Loss:{running_loss}"
    )
    if USE_WANDB:
        wandb.log({"Train Accuracy": accuracy, "epoch": epoch})
    return accuracy


def evaluate(dev_loader, model, criterion, optimizer, epoch):
    with torch.no_grad():
        model.eval()
        all_outputs = []
        all_targets = []
        for i, (input, target) in tqdm(enumerate(dev_loader), total=len(dev_loader)):
            if torch.cuda.is_available():
                input = input.type(torch.FloatTensor).cuda()
                target = target.type(torch.FloatTensor).cuda()

            output = model(input)
            output = output.reshape(-1, 2)
            # loss = criterion(output, target.long())
            output = F.softmax(output, dim=1)
            pred_ans_idx = torch.argmax(output, dim=1)
            all_outputs += list(pred_ans_idx.cpu().numpy())
            all_targets += list(target.cpu().numpy())
            del input
            del target
        accuracy = accuracy_score(all_targets, all_outputs)
        logging.info(f"Epoch: {epoch} | Val Accuracy: {accuracy}")
        if USE_WANDB:
            wandb.log({"Val Accuracy": accuracy, "epoch": epoch})
    return accuracy


def train_and_evaluate_model(
    real_train_npy, adv_train_npy, real_dev_npy, adv_dev_npy, level
):
    num_epochs = params["epochs"]
    BATCH_SIZE = params["batch_size"]
    lr = params["lr"]

    train_dataset = AdvDataset(real_train_npy, adv_train_npy)
    dev_dataset = AdvDataset(real_dev_npy, adv_dev_npy)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE)
    # Level 1(c1).shape = ([N, 16, 32, 32])
    # Level 2(c2).shape = ([N, 16, 32, 32])
    # Level 3(c3).shape = ([N, 32, 16, 16])
    # Level 4(c4).shape = ([N, 64, 8, 8])
    # levels correspond to levels in https://arxiv.org/abs/1702.04267
    level_config_map = {
        1: {"in_channels": 16, "pool1": False, "pool2": False},
        2: {"in_channels": 16, "pool1": False, "pool2": False},
        3: {"in_channels": 32, "pool1": False, "pool2": False},
        4: {"in_channels": 64, "pool1": False, "pool2": False},
    }
    if level not in level_config_map:
        raise Exception(
            f"Invalid Level number {level}, must be one of {list(level_config_map.keys())}"
        )
    else:
        in_channels = level_config_map[level]["in_channels"]
        pool1 = level_config_map[level]["pool1"]
        pool2 = level_config_map[level]["pool2"]
        model = AdvDetector(in_channels, pooling1=pool1, pooling2=pool2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    if torch.cuda.is_available():
        model = model.cuda()
    for epoch in range(num_epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        evaluate(dev_loader, model, criterion, optimizer, epoch)
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"adv_det_level{level}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), f"adv_det_level{level}_epoch_{epoch}.pth")
    wandb.save(f"adv_det_level{level}_epoch_{epoch}.pth")


if __name__ == "__main__":

    train_and_evaluate_model(
        f"./data/real/c{adv_net_level}s_train.npy",
        f"./data/adversarial/c{adv_net_level}s_train.npy",
        f"./data/real/c{adv_net_level}s_dev.npy",
        f"./data/adversarial/c{adv_net_level}s_dev.npy",
        level=adv_net_level,
    )

