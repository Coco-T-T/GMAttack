import torch
import torch.nn.functional as F
from enum import Enum
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

class NormType(Enum):
    Linf = 0
    L2 = 1

def clamp_by_l2(x, max_norm):
    norm = torch.norm(x, dim=(1,2,3), p=2, keepdim=True)
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    return x * factor

def random_init(x, norm_type, epsilon):
    delta = torch.zeros_like(x)
    if norm_type == NormType.Linf:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data * epsilon
    elif norm_type == NormType.L2:
        delta.data.uniform_(0.0, 1.0)
        delta.data = delta.data - x
        delta.data = clamp_by_l2(delta.data, epsilon)
    return delta

def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel


class myImageAttacker():
    # PGD
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True, *args, **kwargs):
        self.norm_type = norm_type
        self.random_init = random_init
        self.epsilon = epsilon
        self.cls = cls
        self.preprocess = kwargs.get('preprocess')
        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (0, 1)

    def input_diversity(self, image):
        return image

    def attack(self, image, masks, num_iters):
        if self.random_init:
            self.delta = random_init(image, self.norm_type, self.epsilon)
        else:
            self.delta = torch.zeros_like(image)

        if hasattr(self, 'kernel'):
            self.kernel = self.kernel.to(image.device)

        if hasattr(self, 'grad'):
            self.grad = torch.zeros_like(image)


        epsilon_per_iter = self.epsilon / num_iters * 1.25

        for i in range(num_iters):
            self.delta = self.delta.detach()
            self.delta.requires_grad = True

            image_diversity = self.input_diversity(image + self.delta)
            #plt.imshow(image_diversity.cpu().detach().numpy()[0].transpose(1, 2, 0))
            if self.preprocess is not None:
                image_diversity = self.preprocess(image_diversity)

            yield image_diversity

            grad = self.get_grad()
            grad = self.normalize(grad)
            self.delta = self.delta.data + epsilon_per_iter * grad
            
            self.delta = self.delta * masks

            # constraint 1: epsilon
            self.delta = self.project(self.delta, self.epsilon)
            # constraint 2: image range
            self.delta = torch.clamp(image + self.delta, *self.bounding) - image

        yield (image + self.delta).detach()

    def get_grad(self):
        self.grad = self.delta.grad.clone()
        return self.grad

    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)

    def normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)

    def run_trades(self, net, image, num_iters):
        with torch.no_grad():
            origin_output = net.inference_image(self.preprocess(image))
            if self.cls:
                origin_embed = origin_output['image_embed'][:, 0, :].detach()
            else:
                origin_embed = origin_output['image_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        attacker = self.attack(image, num_iters)

        for i in range(num_iters):
            image_adv = next(attacker)
            adv_output = net.inference_image(image_adv)
            if self.cls:
                adv_embed = adv_output['image_embed'][:, 0, :]
            else:
                adv_embed = adv_output['image_embed'].flatten(1)

            loss = criterion(adv_embed.log_softmax(dim=-1), origin_embed.softmax(dim=-1))
            loss.backward()

        image_adv = next(attacker)
        return image_adv


class ImageAttacker():
    # PGD
    def __init__(self, epsilon, norm_type=NormType.Linf, random_init=True, cls=True, *args, **kwargs):
        self.norm_type = norm_type
        self.random_init = random_init
        self.epsilon = epsilon
        self.cls = cls
        self.preprocess = kwargs.get('preprocess')
        self.bounding = kwargs.get('bounding')
        if self.bounding is None:
            self.bounding = (0, 1)

    def input_diversity(self, image):
        return image

    def attack(self, image, num_iters):
        if self.random_init:
            self.delta = random_init(image, self.norm_type, self.epsilon)
        else:
            self.delta = torch.zeros_like(image)

        if hasattr(self, 'kernel'):
            self.kernel = self.kernel.to(image.device)

        if hasattr(self, 'grad'):
            self.grad = torch.zeros_like(image)


        epsilon_per_iter = self.epsilon / num_iters * 1.25

        for i in range(num_iters):
            self.delta = self.delta.detach()
            self.delta.requires_grad = True

            image_diversity = self.input_diversity(image + self.delta)
            #plt.imshow(image_diversity.cpu().detach().numpy()[0].transpose(1, 2, 0))
            if self.preprocess is not None:
                image_diversity = self.preprocess(image_diversity)

            yield image_diversity

            grad = self.get_grad()
            grad = self.normalize(grad)
            self.delta = self.delta.data + epsilon_per_iter * grad

            # constraint 1: epsilon
            self.delta = self.project(self.delta, self.epsilon)
            # constraint 2: image range
            self.delta = torch.clamp(image + self.delta, *self.bounding) - image

        yield (image + self.delta).detach()

    def get_grad(self):
        self.grad = self.delta.grad.clone()
        return self.grad

    def project(self, delta, epsilon):
        if self.norm_type == NormType.Linf:
            return torch.clamp(delta, -epsilon, epsilon)
        elif self.norm_type == NormType.L2:
            return clamp_by_l2(delta, epsilon)

    def normalize(self, grad):
        if self.norm_type == NormType.Linf:
            return torch.sign(grad)
        elif self.norm_type == NormType.L2:
            return grad / torch.norm(grad, dim=(1, 2, 3), p=2, keepdim=True)

    def run_trades(self, net, image, num_iters):
        with torch.no_grad():
            origin_output = net.inference_image(self.preprocess(image))
            if self.cls:
                origin_embed = origin_output['image_embed'][:, 0, :].detach()
            else:
                origin_embed = origin_output['image_embed'].flatten(1).detach()

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        attacker = self.attack(image, num_iters)

        for i in range(num_iters):
            image_adv = next(attacker)
            adv_output = net.inference_image(image_adv)
            if self.cls:
                adv_embed = adv_output['image_embed'][:, 0, :]
            else:
                adv_embed = adv_output['image_embed'].flatten(1)

            loss = criterion(adv_embed.log_softmax(dim=-1), origin_embed.softmax(dim=-1))
            loss.backward()

        image_adv = next(attacker)
        return image_adv