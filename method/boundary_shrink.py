import copy
import torch
from torch import nn

from method._adv_generator import LinfPGD, FGSM
from method._expand_exp import curvature, weight_assign

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm

@timer
def boundary_shrink(ori_model, train_forget_loader,
                    unlearn_epoch, unlearn_rate, # wo_BN,
                    logger, console_handler,
                    loader_dict, experiment_path, eval_opt = eval_opt, disable_bn = False,
                    ##################### 一些不明就里的BS参数，没有变动 #########################
                    bound=0.1, step=8 / 255, iter=5,
                    extra_exp=None, lambda_=0.7, bias=-0.5, slope=5.0,
                    ):
                    #########################################################################
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    norm = True  
    random_start = False 

    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    test_model = copy.deepcopy(ori_model).to("cuda")

    # adv = LinfPGD(test_model, bound, step, iter, norm, random_start, "cuda")
    adv = FGSM(test_model, bound, norm, random_start, "cuda")
    # forget_data_gen = inf_generator(train_forget_loader)    # 包装成一个无穷的迭代器
    # batches_per_epoch = len(train_forget_loader)
    # len(loader)是batch的数量，loader.dataset.data.shape中是样本的数量

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)

    num_hits = 0
    num_sum = 0
    # nearest_label = []

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
            x_adv = adv.perturb(x, y, target_y=None, model=test_model, device="cuda")   # 在干净的模型上训练样本
            adv_logits = test_model(x_adv)
            pred_label = torch.argmax(adv_logits, dim=1)
            num_hits += (y != pred_label).float().sum() # 成功攻击的样本数
            num_sum += y.shape[0]

            # INFO: 需要在epoch循环中设置，因为每个epoch的test都会设置为eval
            unlearn_model.train()
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            unlearn_model.zero_grad()
            optimizer.zero_grad()   # 注意清空了模型的梯度和优化器的梯度，保证万无一失

            ori_logits = unlearn_model(x)
            ori_loss = criterion(ori_logits, pred_label)

            if extra_exp == 'curv':
                ori_curv = curvature(ori_model, x, y, h=0.9)[1]
                cur_curv = curvature(unlearn_model, x, y, h=0.9)[1]
                delta_curv = torch.norm(ori_curv - cur_curv, p=2)
                loss = ori_loss + lambda_ * delta_curv  # - KL_div
            elif extra_exp == 'weight_assign':
                weight = weight_assign(adv_logits, pred_label, bias=bias, slope=slope)
                ori_loss = (torch.nn.functional.cross_entropy(ori_logits, pred_label, reduction='none') * weight).mean()
                loss = ori_loss
            else:
                loss = ori_loss  # - KL_div
            loss.backward()
            optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")
        
        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)
    
    log_utils.enable_console_logging(logger, console_handler, True)
    logger.info(f'attack success ratio:  {(num_hits / num_sum).item():.4f}' ) 
    return unlearn_model