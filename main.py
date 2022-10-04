import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import argparse
import os
import numpy as np
from pprint import pprint

from utils import set_seed, PoisonDataset, make_and_restore_cifar_model
from utils import infer_poison_name, infer_exp_name
from train import train_model, eval_model


def make_data(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.ToTensor()

    if args.poison_type == 'Clean':
        train_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        val_set = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_test)
    else:
        train_set = PoisonDataset(args.poison_path, transform=transform_train)
        val_set = PoisonDataset(args.poison_path, transform=transform_test)
    
    set_seed(args.seed)
    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[args.val_num_examples:], indices[:args.val_num_examples]

    train_set = Subset(train_set, train_idx)
    val_set = Subset(val_set, valid_idx)

    test_set = datasets.CIFAR10(args.data_path, train=False, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader, test_loader

def main(args):
    train_loader, val_loader, test_loader = make_data(args)
    set_seed(args.seed)
    if not os.path.isfile(args.model_path):
        model = make_and_restore_cifar_model(args.arch)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_step)
        writer = SummaryWriter(args.tensorboard_path)
        train_model(args, model, optimizer, schedule, train_loader, val_loader, test_loader, writer)
    
    model = make_and_restore_cifar_model(args.arch, resume_path=args.model_path)
    eval_model(args, model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training classifiers for CIFAR-10')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--out_dir', default='results/CIFAR10', type=str)

    parser.add_argument('--train_loss', default='ST', type=str, choices=['ST', 'AT', 'TRADES', 'MART'])
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--arch', default='ResNet18', type=str, choices=['VGG16', 'ResNet18', 'DenseNet121', 'WRN28-10', 'GoogLeNet', 'MobileNetV2'])
    parser.add_argument('--constraint', default='Linf', type=str, choices=['Linf', 'L2'])

    parser.add_argument('--poison_type', default='Clean', type=str, choices=['Clean', 'Adv', 'Hyp'])
    parser.add_argument('--poison_steps', default=100, type=int)
    parser.add_argument('--poison_aug', default=True, type=bool)
    parser.add_argument('--craft_model_loss', default='AT', type=str, choices=['ST', 'AT'])
    parser.add_argument('--craft_model_eps', default=2, type=float)
    parser.add_argument('--craft_model_epoch', default=10, type=int)
    parser.add_argument('--craft_model_arch', default='ResNet18', type=str)
    
    args = parser.parse_args()
    
    # Training options
    args.lr_milestones = [75, 90]
    args.batch_size = 128
    args.lr = 0.1
    args.lr_step = 0.1
    args.weight_decay = 5e-4
    args.val_num_examples = 1000
    args.log_gap = 5
    if args.train_loss == 'MART':
        args.lr = 0.05
    
    # Attack options
    args.eps = args.eps / 255
    args.craft_model_eps = args.craft_model_eps / 255
    args.step_size = args.eps / 4
    args.num_steps = 10
    args.random_restarts = 1

    # Miscellaneous
    args.data_path = '../datasets/CIFAR10'
    args.poison_name = infer_poison_name(args.poison_type, args.poison_steps, args.craft_model_loss, args.craft_model_eps, args.craft_model_epoch, args.craft_model_arch, args.poison_aug)
    args.exp_name = infer_exp_name(args.train_loss, args.eps, args.epochs, args.arch, args.poison_name, args.seed)
    args.tensorboard_path = os.path.join(args.out_dir, args.exp_name, 'tensorboard')
    args.model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pth')
    args.model_path_last = os.path.join(args.out_dir, args.exp_name, 'checkpoint_last.pth')

    args.craft_model_exp_name = infer_exp_name(args.craft_model_loss, args.craft_model_eps, args.craft_model_epoch, args.craft_model_arch, 'Clean')
    args.poison_path = os.path.join(args.out_dir, args.craft_model_exp_name, args.poison_name + '.poison')

    pprint(vars(args))

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    main(args)
    