import copy
import torch
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)  # 计算所有grad向量cat后的l1范数

@timer
def finetune(   ori_model, train_remain_loader,  # 只在train remain上进行训练
                unlearn_epoch, unlearn_rate,
                logger, console_handler,
                loader_dict, experiment_path,
                eval_opt = eval_opt,
                ############## 一些salun中的参数
                with_l1=False, no_l1_epochs=float("inf"), alpha=0):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    # remain_data_gen = inf_generator(train_remain_loader)
    # batches_per_epoch = len(train_remain_loader)

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
        for x, y in train_remain_loader:
            x, y = x.to("cuda"), y.to("cuda")

            unlearn_model.train()
            optimizer.zero_grad()

            if with_l1:
                if epoch < unlearn_epoch - no_l1_epochs:
                    current_alpha = alpha * (
                        1 - epoch / (unlearn_epoch - no_l1_epochs)
                    )
                else:
                    current_alpha = 0
        
            logits = unlearn_model(x)
            loss = criterion(logits, y)
            if with_l1:
                loss += current_alpha * l1_regularization(unlearn_model)

            loss.backward()
            optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")
        
        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)
    
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model