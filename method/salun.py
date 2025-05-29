import copy
import torch
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm

def noise_label(label, num_classes, approx_different):
    # NOTE:也可以用于多类遗忘，但是产生的标签可能是遗忘类别中的
    is_label_lst = isinstance(label, list)

    label = torch.tensor(label) if is_label_lst else label
    if approx_different:
        noisy_label = torch.randint(0, num_classes, (len(label),)).long()
    else:
        shift = torch.randint(1, num_classes, (len(label),)).long()
        noisy_label = (label+shift)%num_classes

    return noisy_label.tolist() if is_label_lst else noisy_label


def gradient_saliency_mask(model, forget_loader, threshold_ratio):
    optimizer = torch.optim.SGD(model.parameters(), 0.1)
    criterion = nn.CrossEntropyLoss() # 默认设置
    # criterion = nn.CrossEntropyLoss(reduction='sum') 

    model.eval()

    gradients = {}
    for name, param in model.named_parameters():
        # gradients[name] = 0 # 后面在+=会自动广播。但我明明记得有时不能广播？
        gradients[name] = torch.zeros_like(param)

    for image, target in tqdm.tqdm(forget_loader, desc="生成mask"):
        image = image.cuda()
        target = target.cuda()

        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():  # NOTE:记录梯度不需要记录计算图，必须关闭
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data
                else:
                    print(f"{name} has no grad")

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])


    # sorted_dict_positions = {}  # 根据参数的梯度大小进行的，梯度大的元素排名靠前
    # 梯度的绝对值大意味着对应的模型参数对损失函数的影响较大，即这个参数的微小变化会导致损失函数的值有较大的变化。这通常意味着这个参数在模型中起着重要的作用
    mask_dict = {}  # 根据阈值进行的，大于阈值的元素为1，小于阈值的元素为0

    # Concatenate all tensors into a single tensor
    all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()]) # 梯度越大，对应的元素越小

    # Calculate the threshold index for the top 10% elements
    threshold_index = int(len(all_elements) * threshold_ratio)    # len(a) 相当于 a.shape[0]

    # Calculate positions of all elements
    positions = torch.argsort(all_elements) # 从小到大排序，返回的是排序后的索引
    ranks = torch.argsort(positions)        # 逆映射，表示 all_elements 中每个元素在原始数组中的排名。从小到大的排序

    # positions 是 all_elements 中元素的排序索引，表示如果将 all_elements 从小到大排序，每个元素在排序后的位置。
    # ranks 是 positions 的逆映射，表示 all_elements 中每个元素在原始数组中的排名。例如，ranks 中的第一个元素表示 all_elements 中第一个元素在排序后的排名。

    start_index = 0
    for key, tensor in gradients.items():
        num_elements = tensor.numel()   # 返回张量中的元素总个数
        # tensor_positions = positions[start_index: start_index + num_elements]
        tensor_ranks = ranks[start_index : start_index + num_elements]

        # sorted_positions = tensor_ranks.reshape(tensor.shape)
        # sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1    # 由于对梯度取了负数，所以mask的是对forget重要的参数
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        mask_dict[key] = threshold_tensor
        start_index += num_elements
    return mask_dict


@timer
def salun(   ori_model, train_forget_loader, num_classes,
                    unlearn_epoch, unlearn_rate,
                    fixed_noise_label, 
                    logger, console_handler,
                    loader_dict, experiment_path,
                    eval_opt = eval_opt,
                    threshold_ratio=0.1,
                    approx_different = True,
                    retain_data=False,
                    mask=True,
                    disable_bn = False, 
                    ):
    mask_dict = gradient_saliency_mask(ori_model, train_forget_loader, threshold_ratio)
    # mask_dict_v2 = gradient_saliency_mask(ori_model, train_forget_loader, threshold_ratio, negative_loss=True)
    # for _, mask in mask_dict.items():
    #     mask_v2 = mask_dict_v2[_]
    #     print(_, torch.sum(mask != mask_v2))
    #     assert  torch.sum(mask != mask_v2)<10, f"{_} param mask: {torch.sum(mask.float())}!={torch.sum(mask_v2.float())}"
    # # 三个因素会导致不同：celoss是mean还是sum，bs是不是1，是否采用数据增广。最后的影响最大，达到几百个，前者影响较小
    # 实验表明negative不影响。
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    # test_model = copy.deepcopy(ori_model).to("cuda")    # 测试random label的成功率
    unlearn_model = copy.deepcopy(ori_model).to("cuda")

    train_forget_loader_randlabel = copy.deepcopy(train_forget_loader)    # INFO: 外层可能用到，因此不能改变，需要深拷贝
    # if fixed_noise_label:
    #     note_print("使用固定噪声标签")
    # else:
    #     note_print("使用随机噪声标签")
    
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
            if not fixed_noise_label:
                y = noise_label(y, num_classes, approx_different)
            x, y = x.to("cuda"), y.to("cuda")
            unlearn_model.train()   # NOTE: 注意unlearn model也被设置为了train，这可能影响bn。在unlearn中是不是应该设置为eval？
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval() 
            unlearn_model.zero_grad()
            optimizer.zero_grad()   # 注意清空了模型的梯度和优化器的梯度，保证万无一失

            logits = unlearn_model(x)
            loss = criterion(logits, y)

            loss.backward()
            if mask:
                for name, param in unlearn_model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask_dict[name]
            optimizer.step()

        if retain_data: # follow this implementation from https://github.com/OPTML-Group/Unlearn-Saliency/blob/master/Classification/unlearn/RL.py
            for x, y in loader_dict["train_remain"]:
                x, y = x.to("cuda"), y.to("cuda")

                unlearn_model.train()  
                if disable_bn:
                    for module in unlearn_model.modules():
                        if isinstance(module, nn.BatchNorm2d):
                            module.eval() 
                unlearn_model.zero_grad()
                optimizer.zero_grad()

                logits = unlearn_model(x)
                loss = criterion(logits, y)

                loss.backward()
                if mask:
                    for name, param in unlearn_model.named_parameters():
                        if param.grad is not None:
                            param.grad *= mask_dict[name]
                optimizer.step()

        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)
    
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model