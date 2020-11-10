# 2 training loops

# BaseModel and AD model

import torch
import torch.nn as nn
import base_model
import logging
from sklearn.metrics import accuracy_score
import torchvision.datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np

params = {}

params["round_name"] = "round0"
params["batch_size"] = 128  # 256 * 3
params["epochs"] = 80


def main():
    BATCH_SIZE = params["batch_size"]
    num_epochs = params["epochs"]

    logging.info("Loading datasets...")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6
    )  # remember to disable shuffle as we are storing data for other models to use

    dev_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=6
    )

    model = nn.DataParallel(base_model.BaseModel(num_classes=10))

    if torch.cuda.is_available():
        model = model.cuda()
    MODEL_FILE = "./savedmodels/epoch_56.pth"
    model.load_state_dict(torch.load(MODEL_FILE))

    val_data_results = run_model(dev_loader, model)
    save_results(val_data_results, "dev")
    train_data_results = run_model(train_loader, model)
    save_results(train_data_results, "train")


def save_results(data_results, tag):
    """Save numpy arrays to disk

    Args:
        data_results (tuple): tuple containing the various(check line1 of fn) np arrays to save
        tag (str): name to suffix while saving the arrays
    """
    outputs, c1s, c2s, c3s, c4s, targets = data_results
    filetag = f"{tag}.npy"
    np.save(f"outputs_{filetag}", outputs)
    np.save(f"c1s_{filetag}", c1s)
    np.save(f"c2s_{filetag}", c2s)
    np.save(f"c3s_{filetag}", c3s)
    np.save(f"c4s_{filetag}", c4s)
    np.save(f"targets_{filetag}", targets)


def run_model(data_loader, model, layer_outputs=True):
    outputs = []
    c1s = []
    c2s = []
    c3s = []
    c4s = []
    targets = []
    with torch.no_grad():
        model.eval()
        all_outputs = []
        all_targets = []
        for i, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader)):
            if torch.cuda.is_available():
                input = input.type(torch.FloatTensor).cuda()
                target = target.type(torch.FloatTensor).cuda()

            if layer_outputs:
                output, c1, c2, c3, c4 = model(input, layer_outputs)
                c1s.append(c1.detach().cpu())
                c2s.append(c2.detach().cpu())
                c3s.append(c3.detach().cpu())
                c4s.append(c4.detach().cpu())
            else:
                output = model(input)
            outputs.append(output.detach().cpu())
            targets.append(target.detach().cpu())
            # loss.append(los)iterion(output, target.long())
            # pred_ans_idx = torch.argmax(output, axis=1)
            # all_outputs += list(pred_ans_idx.cpu().numpy())
            # all_targets += list(target.cpu().numpy())
            del input
            del target
    outputs = torch.cat(outputs, dim=0).numpy()
    c1s = torch.cat(c1s, dim=0).numpy()
    c2s = torch.cat(c2s, dim=0).numpy()
    c3s = torch.cat(c3s, dim=0).numpy()
    c4s = torch.cat(c4s, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    return outputs, c1s, c2s, c3s, c4s, targets


if __name__ == "__main__":

    main()
