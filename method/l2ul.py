import copy
import torch
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm
import itertools


from advertorch.attacks import L2PGDAttack


def estimate_parameter_importance(data_loader, model, num_samples=0):  # num_samples指定想迭代的数据数量，可以少于整个数据集的大小
    importance = {n: torch.zeros(p.shape).to("cuda") for n, p in model.named_parameters()
                  if p.requires_grad}   # 对例如bn层的参数不进行求导
    
    n_samples_batches = ((num_samples+1) // data_loader.batch_size) if num_samples > 0 \
        else len(data_loader)
        # else (len(data_loader.dataset) // data_loader.batch_size)   # 不能直接用dataset大小，因为forget是实用sampler从原始数据集中采样的，dataset大小和真实使用数据集大小不同
    
    model.eval()
    for images, targets in itertools.islice(data_loader, n_samples_batches): # 如果n_samples_batches比实际大，但是由于islice设计，不会报错
        outputs = model.forward(images.to("cuda"))
        loss = torch.norm(outputs, p=2, dim=1).mean()   # dim指定维度
        # optimizer.zero_grad()
        model.zero_grad()
        loss.backward()
        with torch.no_grad():  # 使用no_grad避免追踪梯度
            for n, p in model.named_parameters():
                if p.grad is not None:
                    importance[n] += p.grad.abs() * len(targets)

    n_samples = n_samples_batches * data_loader.batch_size   # num_samples可能为0，根据实际情况校正
    importance = {n: (p / n_samples) for n, p in importance.items()}
    return importance

def adv_attack(model, adversary, data, target, num_classes):
    model.eval()
    
    data, target = data.to("cuda"), target.to("cuda")

    attack_label = torch.rand(data.shape[0]).cuda() * num_classes
    attack_label = attack_label.to(torch.long)
    # attack_label = torch.where(attack_label == target, (torch.rand(data.shape[0]).long().cuda()*num_classes + num_classes) // 2, attack_label)
    # maybe erro in source code https://github.com/csm9493/L2UL

    adv_example = adversary.perturb(data, attack_label)

    # inputs_numpy = adv_example.detach().cpu().numpy()
    # labels_numpy = attack_label.cpu().numpy()
    inputs_numpy = adv_example.detach()
    labels_numpy = attack_label

    return inputs_numpy, labels_numpy

@timer
def l2ul_adv(   ori_model, train_forget_loader, num_classes,
                    unlearn_epoch, unlearn_rate,
                    logger, console_handler,
                    loader_dict, experiment_path,
                    eval_opt = eval_opt,
                    adv_eps=0.4, adv_lambda=0.1, reg_lambda=0, disable_bn=False
                    ):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    test_model = copy.deepcopy(ori_model).to("cuda")
    unlearn_model = copy.deepcopy(ori_model).to("cuda")

    if reg_lambda != 0:
        origin_params = {n: p.clone().detach() for n, p in unlearn_model.named_parameters() if p.requires_grad}
        importance = estimate_parameter_importance(train_forget_loader, test_model)
        for key in importance.keys():
            importance[key] = (importance[key] - importance[key].min()) / (importance[key].max() - importance[key].min())  # 对重要性进行归一化
            importance[key] = (1 - importance[key])

    test_model.eval()
    adversary = L2PGDAttack(test_model, eps=adv_eps, eps_iter=0.1, nb_iter=10, rand_init=True, targeted=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)

    accs_dict = {
        'train_forget': [],
        'train_remain': [],
        'test_forget': [],
        'test_remain': []
    }

    log_utils.enable_console_logging(logger, console_handler, False)
    for epoch in tqdm.trange(unlearn_epoch):
        for x, y in train_forget_loader:
            x, y = x.to("cuda"), y.to("cuda")
            test_model.eval()
            x_adv, y_adv = adv_attack(test_model, adversary, x, y, num_classes) # 为3, 224, 224

            unlearn_model.train()
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            unlearn_model.zero_grad()
            optimizer.zero_grad()

            logits = unlearn_model(x)
            # NOTE: 必须添加，否则效果非常差
            unlearn_model.eval()
            logits_adv = unlearn_model(x_adv)

            # loss = -criterion(logits, y)
            loss = -criterion(logits, y) + criterion(logits_adv, y_adv)*adv_lambda
            if reg_lambda != 0:
                loss_reg = 0
                for n, p in unlearn_model.named_parameters():
                    if n in importance.keys():
                        loss_reg += torch.sum(importance[n] * (p - origin_params[n]).pow(2)) / 2
                logger.info(f"loss: {loss.item()}, loss_reg: {loss_reg.item()}")
                # print(f"loss: {loss.item()}, loss_reg: {loss_reg.item()}"[])
                loss += loss_reg * reg_lambda

            loss.backward()
            optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)
    
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model