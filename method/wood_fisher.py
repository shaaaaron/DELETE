import copy
import torch
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm

def get_require_grad_params(model: torch.nn.Module, named=False):
    if named:
        raise NotImplementedError
        return [
            (name, param)
            for name, param in model.named_parameters()
            if param.requires_grad
        ]
    else:
        return [param for name, param in model.named_parameters() 
                        if param.requires_grad and not "head.fc" in name]


def sam_grad(model, loss):
    params = []

    for param in get_require_grad_params(model, named=False):
        params.append(param)

    sample_grad = torch.autograd.grad(loss, params, allow_unused=True)
    sample_grad = [x.view(-1) for x in sample_grad]

    return torch.cat(sample_grad)   # 将param中每个浮点数对应的梯度拼凑到一个tensor中


def apply_perturb(model, v):   # v是待添加的扰动
    curr = 0
    for param in get_require_grad_params(model, named=False):   # param是一个可变对象，传递的是引用
        length = param.view(-1).shape[0]
        param.view(-1).data += v[curr : curr + length].data     # 似乎对于param.view(-1)修改会跳过求导，可以对param.data.view(-1)j进行add_原地加。不确定，后面注意一下
        curr += length


def woodfisher(model, train_dl, criterion, v):
    note_print("只能用于cifar10数据集，用于imagenet需要修改N的值")
    model.eval()
    k_vec = torch.clone(v)
    N = 1000    # 是训练集数量的1/50，在大规模数据上完整计算可能非常耗时
    # N = 300000  # 是训练集数量的3/50
    o_vec = None
    for idx, (data, label) in enumerate(tqdm.tqdm(train_dl)):
        model.zero_grad()
        data, label = data.to("cuda"), label.to("cuda")
        output = model(data)

        loss = criterion(output, label)
        sample_grad = sam_grad(model, loss)
        with torch.no_grad():
            if o_vec is None:
                o_vec = torch.clone(sample_grad)
            else:
                tmp = torch.dot(o_vec, sample_grad)
                k_vec -= (torch.dot(k_vec, sample_grad) / (N + tmp)) * o_vec
                o_vec -= (tmp / (N + tmp)) * o_vec
        if idx > N:
            return k_vec
    return k_vec

@timer
def wood_fisher(ori_model, train_forget_loader, train_remain_loader, train_remain_loader_sole, 
                alpha,  # 用于计算扰动，默认为0.2
                retain_data, # 是否使用retain数据
                logger, console_handler,
                loader_dict, experiment_path,
                eval_opt = eval_opt,):
    logger.info(f"eval option {eval_opt}")
    unlearn_model = copy.deepcopy(ori_model).to("cuda")
    criterion = nn.CrossEntropyLoss()

    accs_dict = {
        'train_forget': [],
        'train_remain': [],
        'test_forget': [],
        'test_remain': []
    }


    params = []

    for param in get_require_grad_params(unlearn_model, named=False):
        params.append(param.view(-1))

    forget_grad = torch.zeros_like(torch.cat(params)).to("cuda")
    retain_grad = torch.zeros_like(torch.cat(params)).to("cuda")

    total = 0
    unlearn_model.eval()    # NOTE: 默认是eval模式
    for i, (data, label) in enumerate(tqdm.tqdm(train_forget_loader)):
        unlearn_model.zero_grad()
        real_num = data.shape[0]
        data, label = data.to("cuda"), label.to("cuda")
        output = unlearn_model(data)

        loss = criterion(output, label)
        f_grad = sam_grad(unlearn_model, loss) * real_num
        forget_grad += f_grad
        total += real_num

    total_2 = 0
    if retain_data:
        for i, (data, label) in enumerate(tqdm.tqdm(train_remain_loader)):
            unlearn_model.zero_grad()
            real_num = data.shape[0]
            data, label = data.to("cuda"), label.to("cuda")
            output = unlearn_model(data)

            loss = criterion(output, label)
            r_grad = sam_grad(unlearn_model, loss) * real_num
            retain_grad += r_grad
            total_2 += real_num

    forget_grad /= total + total_2
    if retain_data:
        retain_grad *= total / ((total + total_2) * total_2)    # NOTE: 有点奇怪的加权平均，可以考虑将total+total_2修改为total。这样两种方法相对于除以对应的数量

    perturb = woodfisher(
        unlearn_model,
        train_remain_loader_sole,
        criterion=criterion,
        v=forget_grad - retain_grad,
    ) if retain_data else forget_grad 
    note_print(f"noise is {torch.sum(perturb)} {torch.norm(perturb)}")
    logger.info(f"noise is {torch.sum(perturb)} {torch.norm(perturb)}")
    
    apply_perturb(unlearn_model, alpha * perturb)
    
    cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
    for key in keys:
        accs_dict[key].append(cur_accs_dict[key])

    plot_unlearn_remain_acc_figure(1, accs_dict, experiment_path, plot_type="scatter")

    return unlearn_model