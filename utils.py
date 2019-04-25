from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, grad


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)


def hypersphere(z, radius=1):
    return z * radius / z.norm(p=2, dim=1, keepdim=True)


def exp_mov_avg(Gs, G, alpha=0.999, global_step=999):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(Gs.parameters(), G.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

 
class Progress:
    """Determine the progress parameter of the training given the epoch and the progression in the epoch
    Args:
          ep_per_trans (int): the number of epochs before changing the progress,
          pmax (int): the maximum progress of the training.
          batchSizeList (list): the list of the batchSize to adopt during the training
    """

    def __init__(self, ep_per_trans, pmax, batchSizeList):
        assert ep_per_trans > 0 and isinstance(ep_per_trans, int), 'ep_per_trans must be int >= 1'
        assert pmax >= 0 and isinstance(pmax, int), 'pmax must be int >= 0'
        batchSizeList = list(int(b) for b in batchSizeList)
        assert isinstance(batchSizeList, list) and \
               all(isinstance(x, int) for x in batchSizeList) and \
               all(x > 0 for x in batchSizeList) and \
               len(batchSizeList) == pmax + 1, \
            'batchSizeList must be a list of int > 0 and of length pmax+1'

        self.ep_per_trans = ep_per_trans
        self.pmax = pmax
        self.p = 0
        self.batchSizeList = batchSizeList

    def progress(self, epoch, i, total):
        """Update the progress given the epoch and the iteration of the epoch
        Args:
            epoch (int): current epoch
            i (int): iteration in the epoch
            total (int): total number of iterations in the epoch

            progress(current_epoch, current_iteration_in_an_epoch, number_of_epochs_per_transition)

            alpha = f(progress)
        """
        x = (epoch + i / total) / self.ep_per_trans # 10 ep, 500 iter, 1000 total_iter, 2 ep_per trans -> 5 x
        p = max(int(x / 2), x - ceil(x / 2), 0) # 5 x -> 2 p
        self.p = min(p, self.pmax)
        return self.p

    def resize(self, images):
        """Resize the images  w.r.t the current value of the progress.
        Args:
            images (Variable or Tensor): batch of images to resize
        """
        x = int(ceil(self.p))
        if x >= self.pmax:
            return images
        else:
            return F.adaptive_avg_pool2d(images, 4 * 2 ** x)

    @property
    def batchSize(self):
        """Returns the current batchSize w.r.t the current value of the progress"""
        x = int(ceil(self.p))
        return self.batchSizeList[x]


class GradientPenalty:
    """Computes the gradient penalty as defined in "Improved Training of Wasserstein GANs"
    (https://arxiv.org/abs/1704.00028)
    Args:
        batchSize (int): batch-size used in the training. Must be updated w.r.t the current batchsize
        lambdaGP (float): coefficient of the gradient penalty as defined in the article
        gamma (float): regularization term of the gradient penalty, augment to minimize "ghosts"
    """

    def __init__(self, batchSize, lambdaGP, gamma=1, device='cpu'):
        self.batchSize = batchSize
        self.lambdaGP = lambdaGP
        self.gamma = gamma
        self.device = device

    def __call__(self, netD, real_data, fake_data, progress):
        alpha = torch.rand(self.batchSize, 1, 1, 1, requires_grad=True, device=self.device)
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates, progress)
        # compute gradients w.r.t the interpolated outputs
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].view(self.batchSize, -1)
        gradient_penalty = (((gradients.norm(2, dim=1) - self.gamma) / self.gamma) ** 2).mean() * self.lambdaGP

        return gradient_penalty
