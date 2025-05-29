import copy
import torch
from torch import nn

from .utils import keys, eval_opt, plot_unlearn_remain_acc_figure, evaluate_model_on_all_loaders
from utils import *
from trainer import *
import log_utils
import tqdm

def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)

@timer
def boundary_expand(ori_model, train_forget_loader, 
                    unlearn_epoch, unlearn_rate, num_classes,
                    logger, console_handler,
                    loader_dict, experiment_path,
                    freeze_linear=False, eval_opt = eval_opt, disable_bn=False
                    ):
    logger.info(f"unlearn_epoch {unlearn_epoch}, unlearn_rate {unlearn_rate}")
    logger.info(f"eval option {eval_opt}")
    #FIXME: 是feature和class数量，需要按需进行修改 应该是512和numclasses
    featuer_dim, num_classes = ori_model.fc.in_features, num_classes
    # assert featuer_dim==512, "feature dim should be 512"
    logger.info(f"feature dim {featuer_dim}, num_classes {num_classes}")
    #TODO: 目前支持resnet18
    unlearn_model = copy.deepcopy(ori_model)
    unlearn_model.fc = nn.Linear(featuer_dim, num_classes + 1)
    init_params(unlearn_model.fc)
    unlearn_model = unlearn_model.to("cuda")    # 将fc放在cuda上

    for name, params in ori_model.fc.named_parameters():
        print(f"{name} has been loaded")
        if 'weight' in name:
            unlearn_model.fc.state_dict()['weight'][0:num_classes, ] = ori_model.fc.state_dict()[name][:, ]
        elif 'bias' in name:
            unlearn_model.fc.state_dict()['bias'][0:num_classes, ] = ori_model.fc.state_dict()[name][:, ]

    # FIXME: 进行修改后需要重新冻结fc层
    if freeze_linear:
        for name, param in unlearn_model.named_parameters():
            if "fc" in name:
                print(f"freeze {name}")
                param.requires_grad_(False)

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
            x ,y = x.to("cuda"), y.to("cuda")

            widen_logits = unlearn_model(x)

            # target label
            target_label = torch.ones_like(y, device="cuda")    # 全部设置成虚假类别
            target_label *= num_classes

            # adv_train
            unlearn_model.train()
            if disable_bn:
                for module in unlearn_model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            unlearn_model.zero_grad()
            optimizer.zero_grad()

            loss = criterion(widen_logits, target_label)

            loss.backward()
            optimizer.step()

        #NOTE: 需要修改一下，因为输出的标签数量不一致。
        logger.info(f"epoch {epoch+1} loss {loss.item():.4f}")

        cur_accs_dict = evaluate_model_on_all_loaders(unlearn_model, loader_dict, eval_opt, logger, extra_class=1)
        for key in keys:
            accs_dict[key].append(cur_accs_dict[key])

        plot_unlearn_remain_acc_figure(epoch+1, accs_dict, experiment_path)

    pruned_fc = nn.Linear(featuer_dim, num_classes)
    for name, params in unlearn_model.fc.named_parameters():
        print(f"{name} has been loaded")
        if 'weight' in name:
            pruned_fc.state_dict()['weight'][:, ] = unlearn_model.fc.state_dict()[name][0:num_classes, ]
        elif 'bias' in name:
            pruned_fc.state_dict()['bias'][:, ] = unlearn_model.fc.state_dict()[name][0:num_classes, ]
    unlearn_model.fc = pruned_fc
    unlearn_model = unlearn_model.to("cuda")
    log_utils.enable_console_logging(logger, console_handler, True)

    return unlearn_model