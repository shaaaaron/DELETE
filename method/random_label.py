import copy
import torch
from torch import nn
from torchvision import datasets
from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders

from utils import *
from trainer import *
import log_utils
import tqdm

def noise_label(label, num_classes, approx_different):
    # 如果为list则转化为tensor，如果为tensor则不变
    is_label_lst = isinstance(label, list)
    label = torch.tensor(label) if is_label_lst else label

    if approx_different:
        noisy_label = torch.randint(0, num_classes, (len(label),)).long()
    else:
        shift = torch.randint(1, num_classes, (len(label),)).long()
        noisy_label = (label+shift)%num_classes

    # 保证返回值和输入值类型一致
    return noisy_label.tolist() if is_label_lst else noisy_label    

@timer
def random_label(   ori_model, train_forget_loader, num_classes,
                    unlearn_epoch, unlearn_rate,
                    fixed_noise_label, 
                    logger, console_handler,
                    loader_dict, experiment_path,
                    approx_different = True, eval_opt = eval_opt, disable_bn=False
                    ):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    unlearn_model = copy.deepcopy(ori_model).to("cuda")

    train_forget_loader_randlabel = copy.deepcopy(train_forget_loader)    # INFO: 外层可能用到，因此不能改变，需要深拷贝
    
    if approx_different:
        note_print("使用近似不同的噪声标签")
    else:
        note_print("使用确定不同的噪声标签")

    if fixed_noise_label:   
        if isinstance(train_forget_loader_randlabel.dataset, datasets.ImageFolder):
            note_print(f"imagefolder dataset: type(samples){type(train_forget_loader_randlabel.dataset.samples)}")
            paths, targets = zip(*train_forget_loader_randlabel.dataset.samples) 
            paths, targets = list(paths), list(targets)
            targets = noise_label(targets, num_classes, approx_different)
            train_forget_loader_randlabel.dataset.samples = list(zip(paths, targets))
        elif    isinstance(train_forget_loader_randlabel.dataset, datasets.CIFAR10)\
                or isinstance(train_forget_loader_randlabel.dataset, datasets.CIFAR100)\
                or isinstance(train_forget_loader_randlabel.dataset, vggface_dataset):
            train_forget_loader_randlabel.dataset.targets = noise_label(train_forget_loader_randlabel.dataset.targets, num_classes, approx_different)

    # print(train_forget_loader.dataset.targets == train_forget_loader_randlabel.dataset.targets)    # False
    # batches_per_epoch = len(train_forget_loader_randlabel)
    # len(loader)是batch的数量，loader.dataset.data.shape中是样本的数量

    criterion = nn.CrossEntropyLoss()
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
        for x, y in train_forget_loader_randlabel:
        # for x, y in tqdm.tqdm(train_forget_loader_randlabel):
            if not fixed_noise_label:
                raise NotImplementedError("弃用")
                y = noise_label(y, num_classes, approx_different)

            x, y = x.to("cuda"), y.to("cuda")

            unlearn_model.train()
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval() 
            unlearn_model.zero_grad()
            optimizer.zero_grad()   # 注意清空了模型的梯度和优化器的梯度，保证万无一失

            logits = unlearn_model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

        # TODO: 传入model、loader，输出acc，利用acc、plot
        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)
    
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model