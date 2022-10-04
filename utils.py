import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pprint import pprint
from PIL import Image
import matplotlib.pyplot as plt

from models.resnet import ResNet18
from models.vgg import VGG
from models.densenet import densenet_cifar
from models.wideresnet import WideResNet
from models.googlenet import GoogLeNet
from models.mobilenetv2 import MobileNetV2

cifar10_class = {-1: '', 0: 'airplane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


def infer_poison_name(poison_type, poison_steps, craft_model_loss, craft_model_eps, craft_model_epoch, craft_model_arch, poison_aug):
    poison_name = poison_type
    if poison_type in ['Adv', 'Hyp']:
        poison_name = '{}-{}-{}-e{}-a{}-{}'.format(
            poison_type, 
            poison_steps, 
            'ST' if craft_model_loss == 'ST' else '{}{:.1f}'.format(craft_model_loss, craft_model_eps*255),
            craft_model_epoch, 
            craft_model_arch,
            'w' if poison_aug else 'wo',
            )
    return poison_name

def infer_exp_name(train_loss, eps, epochs, arch, poison_name, seed=0):
    exp_name = '{}-e{}-a{}({}){}'.format(
        train_loss if train_loss == 'ST' else '{}{:.1f}'.format(train_loss, eps*255),
        epochs, 
        arch, 
        poison_name,
        seed,
        )
    return exp_name


def infer_arch(model_path):
    for arch in ['MLP', 'VGG16', 'ResNet18']:
        if arch in model_path:
            return arch

def make_and_restore_cifar_model(arch, resume_path=None):
    if arch == 'ResNet18':
        model = ResNet18()
    elif arch == 'VGG16':
        model = VGG('VGG16')
    elif arch == 'DenseNet121':
        model = densenet_cifar()
    elif arch == 'WRN28-10':
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)
    elif arch == 'GoogLeNet':
        model = GoogLeNet()
    elif arch == 'MobileNetV2':
        model = MobileNetV2()
    model = InputNormalize(model, new_mean=(0.4914, 0.4822, 0.4465), new_std=(0.2471, 0.2435, 0.2616))

    if resume_path is not None:
        print('\n=> Loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        info_keys = ['epoch', 'train_acc', 'cln_val_acc', 'cln_test_acc', 'adv_val_acc', 'adv_test_acc']
        info = {k: checkpoint[k] for k in info_keys}
        pprint(info)
        model.load_state_dict(checkpoint['model'])
    
    model = model.cuda()
    return model

class InputNormalize(nn.Module):
    def __init__(self, model, new_mean=(0.4914, 0.4822, 0.4465), new_std=(0.2471, 0.2435, 0.2616)):
        super(InputNormalize, self).__init__()
        new_mean = torch.tensor(new_mean)[..., None, None]
        new_std = torch.tensor(new_std)[..., None, None]
        self.register_buffer('new_mean', new_mean)
        self.register_buffer('new_std', new_std)
        self.model = model
    def __call__(self, x):
        x = (x - self.new_mean) / self.new_std
        return self.model(x)

class PoisonDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform

        self.data, self.targets = torch.load(self.root)
        self.data = self.data.permute(0, 2, 3, 1)   # convert to HWC
        self.data = (self.data * 255).type(torch.uint8)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax

def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, tcolor=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)                
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                if tcolor:
                    ax.set_title(tlist[h][w], fontsize=fontsize, color=tcolor[h][w])
                else:
                    ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    # plt.show()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy_top1(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct * 100. / target.size(0)

def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k
        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)
        Returns:
            A list of top-k accuracies.
    """
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact

class RandomTransform(torch.nn.Module):
    """Crop the given batch of tensors at a random location.
    Code derived from https://github.com/lhfowl/adversarial_poisons/blob/153f96a7670a85261b4602da76366d94bbc1f1a2/village/materials/diff_data_augmentation.py
    As discussed in https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5
    
    """

    def __init__(self, source_size, target_size, shift=8, fliplr=True, flipud=False, mode='bilinear', align=True):
        """Args: source and target size."""
        super().__init__()
        self.grid = self.build_grid(source_size, target_size)
        self.delta = torch.linspace(0, 1, source_size)[shift]
        self.fliplr = fliplr
        self.flipud = flipud

        self.mode = mode
        self.align = True

    @staticmethod
    def build_grid(source_size, target_size):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        k = float(target_size) / float(source_size)
        direct = torch.linspace(-1, k, target_size).unsqueeze(0).repeat(target_size, 1).unsqueeze(-1)
        full = torch.cat([direct, direct.transpose(1, 0)], dim=2).unsqueeze(0)
        return full

    def random_crop_grid(self, x, randgen=None):
        """https://discuss.pytorch.org/t/cropping-a-minibatch-of-images-each-image-a-bit-differently/12247/5."""
        grid = self.grid.repeat(x.size(0), 1, 1, 1).clone().detach()
        grid = grid.to(device=x.device, dtype=x.dtype)
        if randgen is None:
            randgen = torch.rand(x.shape[0], 4, device=x.device, dtype=x.dtype)

        # Add random shifts by x
        x_shift = (randgen[:, 0] - 0.5) * 2 * self.delta
        grid[:, :, :, 0] = grid[:, :, :, 0] + x_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))
        # Add random shifts by y
        y_shift = (randgen[:, 1] - 0.5) * 2 * self.delta
        grid[:, :, :, 1] = grid[:, :, :, 1] + y_shift.unsqueeze(-1).unsqueeze(-1).expand(-1, grid.size(1), grid.size(2))

        if self.fliplr:
            grid[randgen[:, 2] > 0.5, :, :, 0] *= -1
        if self.flipud:
            grid[randgen[:, 3] > 0.5, :, :, 1] *= -1
        return grid


    def forward(self, x, randgen=None):
        # Make a random shift grid for each batch
        grid_shifted = self.random_crop_grid(x, randgen)
        # Sample using grid sample
        return F.grid_sample(x, grid_shifted, align_corners=self.align, mode=self.mode)