# 2 training loops

# BaseModel and AD model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import base_model
import logging
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import wandb

USE_WANDB = True

wandb.init(project="IDL_Proj_baseline")
if USE_WANDB:
    params = wandb.config
else:
    params = {}


params["round_name"] = "round0"
params["batch_size"] = 128  # 256 * 3
params["lr"] = 0.15
params["momentum"] = 0.9
params["epochs"] = 80
params["sched_step_every_n_epochs"] = 2
params["sched_mult_lr_by_gamma_everystep"] = 0.85
tb_writer = SummaryWriter(log_dir=f"runs/{params['round_name']}")
logging.basicConfig(
    level=logging.INFO
)  # ,filename=f"{round_name}log.txt",filemode="w")
wandb.save("models.py")


def main():
    BATCH_SIZE = params["batch_size"]
    LEARNING_RATE = params["lr"]
    MOMENTUM = params["momentum"]
    num_epochs = params["epochs"]
    LOAD_MODEL = False
    sched_step_every_n_epochs = params["sched_step_every_n_epochs"]
    sched_mult_lr_by_gamma_everystep = params["sched_mult_lr_by_gamma_everystep"]

    # MODEL_FILE= "./epoch_0"
    logging.info("Loading datasets...")
    # train_dataset = data.get_train_dataset()
    # dev_dataset = data.get_val_dataset()
    # logging.info("Loaded Datasets")
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=16,
    #     pin_memory=True,
    # )

    # dev_loader = DataLoader(
    #     dev_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=16,
    #     pin_memory=True,
    # )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=6
    )

    dev_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6
    )
    # model = nn.DataParallel(models.PhonemeModel9(context=CONTEXT_SIZE))
    # model = models.BaseModel(num_classes=4000)
    # model = resnet_model.ResnetModel(num_classes=4000)
    model = nn.DataParallel(base_model.BaseModel(num_classes=10))
    # model = nn.DataParallel(models.BaseModel3(num_classes=4000))

    if torch.cuda.is_available():
        model = model.cuda()
    logging.info(model)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_FILE))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=0.001
    )
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,)

    sched = torch.optim.lr_scheduler.StepLR(
        optimizer, sched_step_every_n_epochs, gamma=sched_mult_lr_by_gamma_everystep
    )
    wandb.watch(model, log="all")
    # sched = torch.optim.lr_scheduler.CyclicLR(optimizer,0.0001,0.001,5, cycle_momentum=False)
    logging.info("Starting Training")
    running_loss = 0
    for epoch in range(num_epochs):
        print("Epoch:", epoch, "LR:", sched.get_last_lr())
        wandb.log({"CurrLR": sched.get_last_lr(), "Epoch": epoch})
        train_acc = train(
            train_loader, model, criterion, optimizer, epoch, running_loss
        )
        val_acc = evaluate(dev_loader, model, criterion, optimizer, epoch)
        sched.step()
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), f"epoch_{epoch}.pth")
    wandb.save(f"epoch_{epoch}.pth")


def train(train_loader, model, criterion, optimizer, epoch, running_loss):
    model.train()
    print_frequency = 10000
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
        # embed()
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        # batch_accuracy = calc_accuracy(output, target)
        pred_ans_idx = torch.argmax(output, axis=1)
        all_outputs += list(pred_ans_idx.cpu().numpy())
        all_targets += list(target.cpu().numpy())

        batch_size = input.shape[0]
        loss_agg += loss.item() * batch_size
        running_loss = loss_agg / ((i + 1) * batch_size)
        wandb.log(
            {"loss": loss.item(), "running_loss": running_loss, "epoch": epoch,}
        )

        if i % print_frequency == 0:
            logging.info(
                f"Epoch:{epoch} | Step:{i}/{len(train_loader)} | Loss: {loss.item()}"  # " | Batch Acc:{batch_accuracy}"
            )
        tb_writer.add_scalar("Train/Loss", loss.item(), base_steps + i)
        del input
        del target
        del loss

    accuracy = accuracy_score(all_targets, all_outputs)
    tb_writer.add_scalar("Train/Accuracy", accuracy, epoch)
    logging.info(
        f"Epoch: {epoch} | Train Accuracy: {accuracy} | Running Loss:{running_loss}"
    )
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
            # loss = criterion(output, target.long())
            pred_ans_idx = torch.argmax(output, axis=1)
            all_outputs += list(pred_ans_idx.cpu().numpy())
            all_targets += list(target.cpu().numpy())
            del input
            del target

        # corrects = torch.sum(torch.eq(all_outputs, all_targets))
        # accuracy = corrects.item() / len(all_targets)

        accuracy = accuracy_score(all_targets, all_outputs)
        tb_writer.add_scalar("Val/Accuracy", accuracy, epoch)
        logging.info(f"Epoch: {epoch} | Val Accuracy: {accuracy}")
        wandb.log({"Val Accuracy": accuracy, "epoch": epoch})
    return accuracy


if __name__ == "__main__":

    main()
