# from overrides import overrides
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

class AttackBase(object):
    def __init__(self, model=None, norm=False, discrete=True, device=None):
        self.model = model
        self.norm = norm
        # Normalization are needed for CIFAR10, ImageNet。代码里面确实使用了
        # CIFAR-10 和 ImageNet 这样的数据集包含了大量的图像，这些图像来自不同的来源，可能具有不同的亮度、对比度等。标准化可以使得不同来源的图像在输入模型之前具有一致的统计特性，提高模型的泛化能力。
        if self.norm:
            self.mean = (0.4914, 0.4822, 0.2265)
            self.std = (0.2023, 0.1994, 0.2010)
            # FIXME: mean和std在norm和inv norm的时候他都弄错了
            # mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
            
            #####
        self.discrete = discrete
        self.device = device or torch.device("cuda")
        self.loss(device=self.device)

    def loss(self, custom_loss=None, device=None):
        device = device or self.device
        self.criterion = custom_loss or nn.CrossEntropyLoss()
        self.criterion.to(device)

    def perturb(self, x):
        raise NotImplementedError

    def normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = (y[:, 0, :, :] - self.mean[0]) / self.std[0]
            y[:, 1, :, :] = (y[:, 1, :, :] - self.mean[1]) / self.std[1]
            y[:, 2, :, :] = (y[:, 2, :, :] - self.mean[2]) / self.std[2]
            return y
        return x

    def inverse_normalize(self, x):
        if self.norm:
            y = x.clone().to(x.device)
            y[:, 0, :, :] = y[:, 0, :, :] * self.std[0] + self.mean[0]
            y[:, 1, :, :] = y[:, 1, :, :] * self.std[1] + self.mean[1]
            y[:, 2, :, :] = y[:, 2, :, :] * self.std[2] + self.mean[2]
            return y
        return x

    def discretize(self, x):
        return torch.round(x * 255) / 255

    # Change this name as "projection"
    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)
                # TODO: channel isolation?
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta # 截断到+- bound之间
        x_adv = torch.clamp(x_adv, 0., 1.)  # 阶段到01之间
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)


class FGSM(AttackBase):
    """
    Fast Gradient Sign Method, 快速梯度符号法，基于损失的梯度的一种攻击方法
    """
    def __init__(self, model=None, bound=None, norm=False, random_start=False, discrete=True, device=None, **kwargs):
        super(FGSM, self).__init__(model, norm, discrete, device)
        self.bound = bound  # 允许的最大扰动范围， 0.1
        self.rand = random_start    # 是否在初始扰动上添加随机性, False

    # @overrides
    def perturb(self, x, y, model=None, bound=None, device=None, **kwargs):
        criterion = self.criterion  # CELoss
        model = model or self.model # 如果没有提供（为None）则使用默认值
        bound = bound or self.bound # 0.1
        device = device or self.device

        model.zero_grad()
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, x_nat, bound=bound,
                                 inverse_normalized=True)
            if self.discretize:
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        pred = model(x_adv)
        if criterion.__class__.__name__ == "NLLLoss":
            pred = F.softmax(pred, dim=-1)
        loss = criterion(pred, y)
        loss.backward()

        grad_sign = x_adv.grad.data.detach().sign() # detach从计算图中分离出来
        x_adv = self.inverse_normalize(x_adv) + grad_sign * bound   # 将图片复原然后扰动 # 梯度下降法是减去，攻击是增加
        x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)

        return x_adv.detach()


class LinfPGD(AttackBase):
    def __init__(self, model=None, bound=None, step=None, iters=None, norm=False, random_start=False, discrete=True,
                 device=None, **kwargs):
        super(LinfPGD, self).__init__(model, norm, discrete, device)
        self.bound = bound
        self.step = step
        self.iter = iters
        self.rand = random_start

    # @overrides
    def perturb(self, x, y, target_y=None, model=None, bound=None, step=None, iters=None, x_nat=None, device=None,
                **kwargs):
        criterion = self.criterion
        model = model or self.model
        bound = bound or self.bound
        step = step or self.step
        iters = iters or self.iter
        device = device or self.device

        model.zero_grad()
        if x_nat is None:
            x_nat = self.inverse_normalize(x.detach().clone().to(device))
        else:
            x_nat = self.inverse_normalize(x_nat.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
            if self.discretize:
                x_adv = self.normalize(self.discretize(x_adv)).detach().clone().requires_grad_(True)
            else:
                x_adv = self.normalize(x_adv).detach().clone().requires_grad_(True)

        for i in range(iters):
            adv_pred = model(x_adv)
            ori_pred = model(x)
            delta_pred = adv_pred - ori_pred
            if criterion.__class__.__name__ == "NLLLoss":
                delta_pred = F.log_softmax(delta_pred, dim=-1)
            # loss =   0.1*criterion(pred, target_y) - criterion(pred, original_y)
            if target_y is not None:
                # loss = criterion(adv_pred, y)
                loss = - criterion(delta_pred, target_y)  # + 0.01*criterion(delta_pred, y)
            else:
                loss = criterion(adv_pred, y)
            loss.backward()

            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
            model.zero_grad()

        return x_adv.detach().to(device)


