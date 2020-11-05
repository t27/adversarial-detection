"""Utilities such as argument parsing"""

import argparse
import torch
import numpy as np


def parse_args():
    """
    :return:  args: experiment configs, device: use CUDA or cpu
    """
    parser = argparse.ArgumentParser(description='Traning a fully connected network to classify speech recording phonemes.')

    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Loads dev set in place of train.')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    return args, device


class RunningAverage():
    """
    Keeps running average of metrics so they don't fluctuate.
    """
    def __init__(self, size=5):
        self.arr = np.zeros(size)
        self.size = size
        self.counter = 0

    def add(self, value):
        self.arr[self.counter % self.size] = value
        self.counter += 1
        if self.counter >= self.size:
            self.counter = 0

    def get(self):
        return np.mean(self.arr)

    def reset(self):
        self.arr = 0