import copy
import torch
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm


def badteacher_loss(
    output, labels, good_teacher_logits, bad_teacher_logits, KL_temperature
):
    labels = torch.unsqueeze(labels, dim=1)

    good_teacher_out = F.softmax(good_teacher_logits / KL_temperature, dim=1)
    bad_teacher_out = F.softmax(bad_teacher_logits / KL_temperature, dim=1)

    overall_teacher_out = labels * bad_teacher_out + (1 - labels) * good_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')



@timer
def bad_teacher(ori_model, bad_teacher_model, good_teacher_model, unlearn_loader,    # 不同于train_forget_loader其中有forget/remain数据，并且标签为0/1.
                unlearn_epoch, unlearn_rate,
                logger, console_handler,
                loader_dict, experiment_path,
                KL_temperature = 1, eval_opt = eval_opt, disable_bn=False
                ):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    test_forget_acc, test_remain_acc = test(good_teacher_model, loader_dict["test_forget"])[1], test(good_teacher_model, loader_dict["test_remain"])[1]
    logger.info(f"good teacher acc {test_forget_acc:.2%}, {test_remain_acc:.2%}")
    test_forget_acc, test_remain_acc = test(bad_teacher_model, loader_dict["test_forget"])[1], test(bad_teacher_model, loader_dict["test_remain"])[1]
    logger.info(f"bad teacher acc {test_forget_acc:.2%}, {test_remain_acc:.2%}")

    tmp_forget_loader = copy.deepcopy(loader_dict["train_forget"])
    logger.info("坏老师关于遗忘类别的输出是")
    for x, y in tmp_forget_loader:
        x, y = x.to("cuda"), y.to("cuda")
        logger.info(f"ground truth是{y}")
        logger.info(f"坏老师关于遗忘类别的输出是{bad_teacher_model(x).max(1)[1]}")
        # logger.info(f"好老师关于遗忘类别的输出是{good_teacher_model(x).max(1)[1]}")
        break

    unlearn_model = copy.deepcopy(ori_model).to("cuda")

    optimizer = torch.optim.SGD(unlearn_model.parameters(), lr=unlearn_rate, momentum=0.9)

    accs_dict = {
        'train_forget': [],
        'train_remain': [],
        'test_forget': [],
        'test_remain': []
    }

    # 关闭控制台日志输出，避免影响tqdm
    log_utils.enable_console_logging(logger, console_handler, False)
    for epoch in tqdm.trange(unlearn_epoch):
        for x, y in unlearn_loader:
            x, y = x.to("cuda"), y.to("cuda")  

            unlearn_model.train()
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            # unlearn_model.zero_grad()
            optimizer.zero_grad()

            logits = unlearn_model(x)

            good_teacher_model.eval()
            bad_teacher_model.eval()

            # this code run slowly
            # with torch.no_grad():
            #     good_teacher_logits = good_teacher_model(x)
            #     bad_teacher_logits = bad_teacher_model(x)

            # loss = badteacher_loss(logits, y, good_teacher_logits, bad_teacher_logits, KL_temperature)

            # accelate by mask, good teacher and bad teacher only forward once on each data
            mask = (y==1)
            bad_teacher_input = x[mask]
            good_teacher_input = x[~mask]

            with torch.no_grad():
                if mask.sum()>0:    # 空向量输入swin-transformer会报错
                    bad_teacher_logits = bad_teacher_model(bad_teacher_input)
                else:
                    bad_teacher_logits = torch.empty_like(logits[0:0])

                if (~mask).sum()>0:
                    good_teacher_logits = good_teacher_model(good_teacher_input)
                else:
                    good_teacher_logits = torch.empty_like(logits[0:0])

            teacher_logits = torch.zeros_like(logits)
            teacher_logits[mask] = bad_teacher_logits
            teacher_logits[~mask] = good_teacher_logits

            teacher_out = F.softmax(teacher_logits / KL_temperature, dim=1)
            student_out = F.log_softmax(logits / KL_temperature, dim=1)
            loss =  F.kl_div(student_out, teacher_out, reduction='batchmean')
            loss.backward()
            optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)
    
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model