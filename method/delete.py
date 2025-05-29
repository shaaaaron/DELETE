import copy
import torch
from torch import nn
import torch.nn.functional as F
import itertools
from collections import OrderedDict

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm



@timer
def delete(   ori_model, train_forget_loader,
                    unlearn_epoch, unlearn_rate,
                    logger, console_handler,
                    loader_dict, experiment_path,
                    soft_label,
                    eval_opt = eval_opt, disable_bn = False,
                    ############## 我的额外自定义参数开始
                    ):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    test_model = copy.deepcopy(ori_model).to("cuda")

    criterion = nn.KLDivLoss(reduction='batchmean') # mean是对所有维度平均，batchmean只对batch维度平均，应该使用后者

    assert soft_label in ["inf"]
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
            unlearn_model.train()

            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval() 

            unlearn_model.zero_grad()
            optimizer.zero_grad()   # 注意清空了模型的梯度和优化器的梯度，保证万无一失

            test_model.eval()
            batch_size = x.shape[0]
            with torch.no_grad():
                pred_label = test_model(x)
            if soft_label == "inf":
                pred_label[torch.arange(batch_size), y] = -1e10
            else:
                raise ValueError("Unknown soft label method")

            ori_logits = unlearn_model(x)

            ori_logits = F.log_softmax(ori_logits, dim=1)   # input log softmax
            pred_label = F.softmax(pred_label, dim=1)       # target softmax
            loss = criterion(ori_logits, pred_label)
            loss.backward()
            optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")
        
        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path) # 实现一个ploter，有横轴、四个纵轴、保存路径和初始化函数。算了感觉用处不大，后面再说
    
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model
