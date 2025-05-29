import copy
import torch
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm

def fisher_information_martix(model, train_dl, device):
    model.eval()
    fisher_approximation = []
    for parameter in model.parameters():
        fisher_approximation.append(torch.zeros_like(parameter).to(device))
    total = 0
    for i, (data, label) in enumerate(tqdm.tqdm(train_dl)):
        data = data.to(device)
        label = label.to(device)
        predictions = torch.log_softmax(model(data), dim=-1)
        real_batch = data.shape[0]

        epsilon = 1e-8
        for i in range(real_batch):
            label_i = label[i]
            prediction = predictions[i][label_i]
            gradient = torch.autograd.grad(
                prediction, model.parameters(), retain_graph=True, create_graph=False
            )
            for j, derivative in enumerate(gradient):
                fisher_approximation[j] += (derivative + epsilon) ** 2
        total += real_batch
    for i, parameter in enumerate(model.parameters()):
        fisher_approximation[i] = fisher_approximation[i] / total

    return fisher_approximation


def hessian(train_remain_loader, model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    for p in model.parameters():
        p.grad2_acc = 0

    for data, orig_target in tqdm.tqdm(train_remain_loader):
        data, orig_target = data.to("cuda"), orig_target.to("cuda")
        output = model(data)
        prob = torch.nn.functional.softmax(output, dim=-1).data

        for y in range(output.shape[1]):
            target = torch.empty_like(orig_target).fill_(y)
            loss = criterion(output, target)
            model.zero_grad()
            loss.backward(retain_graph=True)
            for p in model.parameters():
                if p.requires_grad:
                    p.grad2_acc += torch.mean(prob[:, y]) * p.grad.data.pow(2)

    for p in model.parameters():
        p.grad2_acc /= len(train_remain_loader)


def get_mean_var(p, num_classes, alpha):
    mu = copy.deepcopy(p.data0.clone())

    var = copy.deepcopy(1.0 / (p.grad2_acc + 1e-8))
    var = var.clamp(max=1e3)
    if p.shape[0] == num_classes:
        var = var.clamp(max=1e2)
    var = alpha * var
    if p.ndim > 1:
        var = var.mean(dim=1, keepdim=True).expand_as(p).clone()

    # FIXME: freeze linear需要注释下面你的代码
    if p.shape[0] == num_classes: # Last layer, shape[0]是输出特征维度
        var *= 10
    elif p.ndim == 1: # BatchNorm
        var *= 10
        
    return mu, var

def fisher( ori_model, train_forget_loader, train_remain_loader, 
            alpha, num_classes,
            logger, console_handler,
            loader_dict, experiment_path, eval_opt = eval_opt, freeze_linear = False
            ):
    logger.info(f"unlearn alpha {alpha}")
    logger.info(f"eval option {eval_opt}")
    unlearn_model = copy.deepcopy(ori_model).to("cuda")

    accs_dict = {
        'train_forget': [],
        'train_remain': [],
        'test_forget': [],
        'test_remain': []
    }
    for p in unlearn_model.parameters():
        p.data0 = copy.deepcopy(p.data.clone())

    hessian(train_remain_loader, unlearn_model)
    
    for i, (name, p) in enumerate(unlearn_model.named_parameters()):
        if freeze_linear and "fc" in name:
            continue

        mu, var = get_mean_var(p, num_classes, alpha)
        p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()

    cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
    for key in keys:
        accs_dict[key].append(cur_accs_dict[key])
    plot_unlearn_remain_acc_figure(1, accs_dict, experiment_path, plot_type="scatter")
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model