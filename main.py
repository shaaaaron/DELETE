import argparse
import copy
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime

from utils import *
from trainer import *
import method
import log_utils

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Machine Unlearning")
    ############################################# 方法/数据集/模型 #############################################
    # TODO: 添加其他方法/数据集/模型
    parser.add_argument('--method', type=str, default="boundary_shrink",
                        choices=['random_label', "finetune", "gradient_ascent", 
                                 'boundary_shrink', 'boundary_expand', 
                                 "salun", "l2ul_adv", "l2ul_imp", "bad_teacher",
                                 "fisher", "wood_fisher",
                                 "delete",  # my method
                                 "pass", "ablation"
                                 ], help='unlearning method')
    parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', "cifar100", "tiny_imagenet", "vggface"], help='dataset name')
    # parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['cifar10', "cifar100", "tiny_imagenet"], help='dataset name')
    parser.add_argument('--model_name', type=str, default='resnet18',  choices=['resnet18', "vgg16", "vit-s-16", "swin-t"], help='model name')
    parser.add_argument('--exps_dir', type=str, default="~/boundary_unlearn/classification/exps", help='experiments directory')
    ##########################################################################################################


    ############################################# train from scratch设置 #####################################
    parser.add_argument('--train_from_scratch', action='store_true', help='Train model from scratch')
    parser.add_argument('--retrain_from_scratch', action='store_true', help='Retrain model from scratch')
    parser.add_argument('--debug', action='store_true') # debug 模式跳过一些命令执行
    parser.add_argument('--optim_name', type=str, default='sgd', choices=['sgd', 'adam'], help='optimizer name')
    ##########################################################################################################


    ############################################# lr、unlr 设置，取决于model和dataset ########################
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')   # INFO: 是train from scratch 和 unlearn 的batch size
    parser.add_argument('--pretrain_epoch', type=int, default=None, help='train from scratch epoch')
    parser.add_argument('--pretrain_lr',    type=float, default=None , help='learning rate')
    parser.add_argument('--unlearn_epoch',  type=int, default=None, help='unlearning epoch')
    parser.add_argument('--unlearn_rate',   type=float, default=None) # 是真正的遗忘学习率
    parser.add_argument('--finetune_epoch', type=int, default=None)
    parser.add_argument('--finetune_lr',    type=float, default=None)
    ##########################################################################################################


    ############################################# 遗忘任务设置 #################################################
    # TODO: 添加-1表示all forget class的功能，或者添加其他的遗忘函数，不要写死在一个里面。可以指定比例
    parser.add_argument('--forget_class', type=int, default=1, help='forget class') # INFO:调整为1
    # FIXME: 对cifar10和cifar100不能使用4，后面可以给cifar10添加一个permute_map
    # parser.add_argument('--forget_class', type=int, default=1, help='forget class') 
    # NOTE:如果只用于cifar10表示类别索引，用于cifar100和tiny_imagenet表示遗忘的类别数目
    ##########################################################################################################

    parser.add_argument("--freeze_linear", action="store_true")  # 实验方法，是否冻结线性层，默认不冻结
    
    ############################################# 默认参数，不需要任何调整 #######################################
    parser.add_argument('--extra_exp', type=str, help='optional extra experiment for boundary shrink',
                        choices=['curv', 'weight_assign', None])
    parser.add_argument("--fixed_noise_label", type=str2bool, default=True) # 用于random label遗忘算法，是否固定随机标签
    parser.add_argument("--approx_different", type=str2bool, default=True)  # 用于random，默认近似不同False
    parser.add_argument("--retain_data", type=str2bool, default=False)
    parser.add_argument("--salun_mask", type=str2bool, default=True)    # 只是用了调试去除mask，salun和random label是不是相同的
    ##########################################################################################################

    parser.add_argument("--alpha", type=float, default=0.2)  # 用于fisher/ wood fwood fisher算法，遗忘的的系数
    parser.add_argument("--threshold_ratio", type=float, default=0.5)
    parser.add_argument("--adv_lambda", type=float, default=0.1)    # 默认0.1
    parser.add_argument("--reg_lambda", type=float, default=0)
    parser.add_argument("--adv_eps", type=float, default=0.4)
    ##########################################################################################################


    ############################################# 蒸馏方法设置 #################################################
    parser.add_argument('--soft_label', type=str, default="inf")
    ##########################################################################################################

    parser.add_argument("--ablation_a", default=None, type=float)
    parser.add_argument("--ablation_t", default=None, type=float)
    parser.add_argument("--wo_dataaug", action="store_true")    # 默认为false，使用dataaug
    parser.add_argument("--description", type=str, default="", help="Description for this run")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument('--load_original_model_path', type=str, default=None)
    parser.add_argument('--load_retrain_model_path', type=str, default=None)
    # NOTE: 临时参数

    args = parser.parse_args()

    config = OmegaConf.load(f'config/{args.dataset_name}_{args.model_name}.yaml')   # 除了args用法，可以通过字典用法访问
    keys = ["pretrain_epoch", "pretrain_lr", "batch_size", "unlearn_epoch", "unlearn_rate"]
    if args.dataset_name == "vggface":
        keys += ["finetune_epoch", "finetune_lr"]

    for key in keys:
        if getattr(args, key) is None:
            setattr(args, key, config[key])

    if any([getattr(args, key) is None for key in keys]):
        raise ValueError(f"some key are not set")

    print(args)

    model_name = args.model_name
    # 在cifar10和其他数据集上，forget_class有相同的含义。 不同版本的代码已经修正
    forget_class = args.forget_class
    num_workers = args.num_workers


    description = f"{args.dataset_name}_{model_name}_forget{forget_class}" # 只用于unlearn方法，pretrain不需要 
    if args.freeze_linear:
        description = "freeze_linear_" + description

    # TODO: 建议把method拿出作为一个文件夹，后面也不需要过多修改
    method_description = f"{args.method}"

    vice_description = f"{args.description}" if args.description else ""    # 是args.description，不是description
    now = datetime.now()
    formatted_time = now.strftime("%m%d-%H:%M:%S")
    vice_description += f"_{formatted_time}"

    seed_torch(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert device.type == 'cuda', 'only support cuda'
    # device 是一个torch.device对象，不能使用device == "cuda"进行判断
    path = Path(args.exps_dir).expanduser()    # 保存日志、结果、模型、配置的根目录
    create_dir(path)

    transform_train, transform_test = get_transforms(args.dataset_name, args.model_name, wo_dataaug=args.wo_dataaug)    # INFO:可以选择是否需要数据增广

    if args.dataset_name == "vggface":
        config_path = 'config/vggface_sample.yaml'
        dir_path = "/mnt/Datasets/vggface2"
        try:
            sample_config = OmegaConf.load(config_path)
        except FileNotFoundError:
            samples = create_dataset(dir_path)
            conf = OmegaConf.create(samples)
            OmegaConf.save(conf, config_path)
            sample_config = OmegaConf.load(config_path)

        pretrain_train_dataset = vggface_dataset(sample_config, transform_train, mode='pretrain', train=True)
        pretrain_test_dataset = vggface_dataset(sample_config, transform_test, mode='pretrain', train=False)
        pretrain_train_loader, pretrain_test_loader = get_dataloader(pretrain_train_dataset, pretrain_test_dataset, args.batch_size, num_workers)

    trainset, testset = get_dataset(args.dataset_name, transform_train, transform_test)
    train_loader, test_loader = get_dataloader(trainset, testset, args.batch_size, num_workers)


    num_classes = max(train_loader.dataset.targets) + 1
    assert forget_class < num_classes, 'forget class must less than num_classes'    # 无论是类别索引或类别数量，都应该小于总类别数

    # if args.dataset_name == "cifar10" or args.dataset_name == "vggface":
    #     forget_class_index = [forget_class]
    # else:
    #     # forget_class_index =  random.sample(range(0, num_classes), forget_class)
    #     permutation_map = getattr(config, "permutation_map")
    #     forget_class_index = permutation_map[:forget_class]
    permutation_map = getattr(config, "permutation_map")
    forget_class_index = permutation_map[:forget_class]
    note_print(f"forget class index: {forget_class_index}")

    num_forget = float("inf")       #  全部用于遗忘
    train_forget_loader, train_remain_loader, test_forget_loader, test_remain_loader, repair_class_loader, \
    train_forget_index, train_remain_index, test_forget_index, test_remain_index \
        = get_unlearn_loader(trainset, testset, forget_class_index, args.batch_size, num_forget, num_workers)

    if args.train_from_scratch or args.retrain_from_scratch:
        ckpt_path = path/ "test_pretrained_model"   # 防止不小心跑的覆盖了原来的模型
        create_dir(ckpt_path)
    else:
        ckpt_path = path / "pretrained_model"

    ori_model, retrain_model = None, None
    if args.train_from_scratch:  # original 和 retrain 不使用description， 其他的unlearn使用
        print('=' * 100)
        print(' ' * 25 + 'train original model from scratch')
        print('=' * 100)

        if args.dataset_name == "vggface":
            ori_model = train_save_model(pretrain_train_loader, pretrain_test_loader, model_name, args.optim_name, args.pretrain_lr,
                                        args.pretrain_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_pretrain_model_{args.description}_{formatted_time}")
            
            ori_model.fc = torch.nn.Linear(ori_model.fc.in_features, 10)
            ori_model = ori_model.to("cuda")

            ori_model = finetune_save_model(train_loader, test_loader, ori_model, args.optim_name, args.finetune_lr,
                            args.finetune_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_original_model_{args.description}_{formatted_time}")
            
        else:
            ori_model = train_save_model(train_loader, test_loader, model_name, args.optim_name, args.pretrain_lr,
                                        args.pretrain_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_original_model_{args.description}_{formatted_time}")
        print('\noriginal model acc:\n', test_each_classes(ori_model, test_loader, num_classes))

    if args.retrain_from_scratch:
        print('=' * 100)
        print(' ' * 25 + 'retrain model from scratch')
        print('=' * 100)
        if args.dataset_name == "vggface":
            retrain_model = train_save_model(pretrain_train_loader, pretrain_test_loader, model_name, args.optim_name, args.pretrain_lr,
                            args.pretrain_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_pretrain_model_{args.description}_{formatted_time}")
            retrain_model.fc = torch.nn.Linear(retrain_model.fc.in_features, 10)
            retrain_model = retrain_model.to("cuda")

            retrain_model = finetune_save_model(train_remain_loader, test_remain_loader, retrain_model, args.optim_name, args.finetune_lr,
                            args.finetune_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_retrain_forget{forget_class}_model_{args.description}_{formatted_time}")
        else:
            retrain_model = train_save_model(train_remain_loader, test_remain_loader, model_name, args.optim_name, args.pretrain_lr, 
                                            args.pretrain_epoch, ckpt_path, f"{args.dataset_name}_{model_name}_retrain_forget{forget_class}_model_{args.description}_{formatted_time}")
        print('\nretrain model acc:\n', test_each_classes(retrain_model, test_loader, num_classes))
    if args.train_from_scratch or args.retrain_from_scratch:
        note_print('train/retrain from scratch done，结束运行')
        exit(1)

    print('=' * 100)
    print(' ' * 25 + 'load original model and retrain model')
    print('=' * 100)
    # 加载测试original model
    if args.load_original_model_path:
        original_model_path = Path(args.load_original_model_path)
    else:
        original_model_path = ckpt_path / f'{args.dataset_name}_{model_name}_original_model.pth'
    note_print(f"load original model from {original_model_path}")

    ori_model = load_model(original_model_path, model_name, num_classes)

    if not args.debug:
        # _, acc = test(ori_model, train_loader)
        # note_print(f"original model的性能是")
        # print(f"train acc:{acc:.2%}")
        # _, acc = test(ori_model, test_loader)
        # print(f"test acc:{acc:.2%}")

        _, acc = test(ori_model, train_forget_loader)
        print(f"forget train acc:{acc:.2%}")
        # FIXME:
        # print(f"remain train has been blocked")
        _, acc = test(ori_model, train_remain_loader)
        print(f"remain train acc:{acc:.2%}")
        _, acc = test(ori_model, test_forget_loader)
        print(f"forget test acc:{acc:.2%}")
        _, acc = test(ori_model, test_remain_loader)
        print(f"remain test acc:{acc:.2%}")
        # print('\noriginal model acc:\n', test_each_classes(ori_model, test_loader, num_classes))


    # 加载测试retrain model
    if args.load_retrain_model_path:
        retrain_model_path = Path(args.load_retrain_model_path)
    else:
        retrain_model_path = ckpt_path / f'{args.dataset_name}_{model_name}_retrain_forget{forget_class}_model.pth'
    note_print(f"load retrain model from {retrain_model_path}")
    # NOTE: 代码中可以不使用retrain_model
    retrain_model = load_model(retrain_model_path, model_name, num_classes)

    if not args.debug:
        _, acc = test(retrain_model, train_forget_loader)
        note_print(f"\nretrain model的性能是")
        print(f"forget train acc:{acc:.2%}")
        # FIXME:
        # print(f"remain train has been blocked")
        _, acc = test(retrain_model, train_remain_loader)
        print(f"remain train acc:{acc:.2%}")
        _, acc = test(retrain_model, test_forget_loader)
        print(f"forget test acc:{acc:.2%}")
        _, acc = test(retrain_model, test_remain_loader)
        print(f"remain test acc:{acc:.2%}")
        # print('\nretrain model acc:\n', test_each_classes(retrain_model, test_loader, num_classes))


    create_dir(path / description)
    create_dir(path / description / method_description)
    create_dir(path / description / method_description / vice_description)
    args_dict = vars(args)
    config = OmegaConf.create(args_dict)
    OmegaConf.save(config, path / description / method_description / vice_description / "config.yaml")
    logger, console_handler = log_utils.setup_logger(path / description / method_description / vice_description, logger_name="train_log")
    log_utils.enable_console_logging(logger, console_handler, True)
    
    unlearn_model = None
    loader_dict = {"train_forget": train_forget_loader, "train_remain": train_remain_loader,
                   "test_forget": test_forget_loader, "test_remain": test_remain_loader,
                   "test": test_loader, 
                   }
    
    print('*' * 100)
    if args.method:
        note_print(' ' * 25 + f'begin {args.method.replace("_", " ")} unlearning')
    print('*' * 100)

    if args.freeze_linear:
        for name, param in ori_model.named_parameters():
            if "fc" in name:
                print(f"freeze {name}")
                param.requires_grad_(False)

    disable_bn = False
    if forget_class == 1 and args.dataset_name == "tiny_imagenet":  # 在iamgenet上进行单类遗忘
        disable_bn = True
        note_print("disable bn for tiny imagenet for single class forget")

    experiment_path = path / description / method_description / vice_description
    if args.method == "random_label":
        unlearn_model = method.random_label(ori_model, train_forget_loader, num_classes,
                                                    args.unlearn_epoch, args.unlearn_rate,
                                                    fixed_noise_label = args.fixed_noise_label,
                                                    logger = logger, console_handler = console_handler, 
                                                    loader_dict=loader_dict, experiment_path = experiment_path, 
                                                    approx_different = args.approx_different, disable_bn = disable_bn)
    elif args.method == "finetune":
        # FIXME:暂时不在tiny imagenet中使用finetune方法
        unlearn_model = method.finetune(ori_model, train_remain_loader, 
                                                unlearn_epoch= args.unlearn_epoch, unlearn_rate= args.unlearn_rate,
                                                logger = logger, console_handler = console_handler, 
                                                loader_dict=loader_dict, experiment_path = experiment_path)  # 存在一些salun实现的finetune专属参数，但是没有使用
    elif args.method == "gradient_ascent":
        unlearn_model = method.gradient_ascent(ori_model, train_forget_loader,
                                                        unlearn_epoch=args.unlearn_epoch, unlearn_rate=args.unlearn_rate,
                                                        logger=logger, console_handler=console_handler,
                                                        loader_dict=loader_dict, experiment_path= experiment_path, disable_bn = disable_bn)
    elif args.method == 'boundary_shrink':
        unlearn_model = method.boundary_shrink( ori_model, train_forget_loader, args.unlearn_epoch, args.unlearn_rate, 
                                                        logger = logger, console_handler = console_handler,
                                                        loader_dict = loader_dict, experiment_path = experiment_path, disable_bn = disable_bn,
                                                        extra_exp=args.extra_exp,
                                                        )
    elif args.method == 'boundary_expand':   # NOTE: 使用resnet18之外的模型可能存在问题，必须具有fc
        unlearn_model = method.boundary_expand( ori_model, train_forget_loader, args.unlearn_epoch, args.unlearn_rate, num_classes,
                                                        logger = logger, console_handler = console_handler,
                                                        loader_dict = loader_dict, experiment_path = experiment_path, disable_bn = disable_bn, 
                                                        freeze_linear = args.freeze_linear  # 赋值
                                                       ) 
    elif args.method == "salun":
        unlearn_model = method.salun(ori_model, train_forget_loader, num_classes,
                                            unlearn_epoch=args.unlearn_epoch, unlearn_rate=args.unlearn_rate,
                                            fixed_noise_label=args.fixed_noise_label, 
                                            logger=logger, console_handler=console_handler,
                                            loader_dict=loader_dict, experiment_path= experiment_path,
                                            threshold_ratio=args.threshold_ratio,
                                            approx_different=args.approx_different,
                                            retain_data=args.retain_data,  disable_bn = disable_bn, 
                                            mask=args.salun_mask)
    elif args.method == "bad_teacher":
        good_teacher_model = copy.deepcopy(ori_model).to("cuda")
        bad_teacher_model = get_model(model_name, num_classes).to("cuda")

        filtered_remain_index = random.sample(train_remain_index, int(0.3*len(train_remain_index))) if args.retain_data else []
        
        class UnLearningData(Dataset):
            def __init__(self, dataset, forget_index, remain_index):
                super().__init__()
                self.dataset = dataset
                self.index = forget_index + remain_index
                
                self.len = len(forget_index) + len(remain_index)
                self.forget_index_len = len(forget_index)

            def __len__(self):
                return self.len

            def __getitem__(self, index):
                mapped_index = self.index[index]
                x = self.dataset[mapped_index][0]
                y = 1 if index < self.forget_index_len else 0
                return x, y


        unlearn_dataset = UnLearningData(trainset, train_forget_index, filtered_remain_index)
        unlearn_loader = torch.utils.data.DataLoader(unlearn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers) 
        unlearn_model = method.bad_teacher(ori_model, bad_teacher_model, good_teacher_model, unlearn_loader,
                                            args.unlearn_epoch, args.unlearn_rate,
                                            logger = logger, console_handler = console_handler, 
                                            loader_dict=loader_dict, experiment_path = experiment_path, disable_bn = disable_bn)
    elif args.method == "l2ul_adv":
        unlearn_model = method.l2ul_adv(ori_model, train_forget_loader, num_classes,
                                            args.unlearn_epoch, args.unlearn_rate,
                                            logger = logger, console_handler = console_handler, 
                                            loader_dict=loader_dict, experiment_path = experiment_path, disable_bn = disable_bn,
                                            adv_eps = args.adv_eps,
                                            adv_lambda=args.adv_lambda)
    elif args.method == "l2ul_imp":
        unlearn_model = method.l2ul_adv(ori_model, train_forget_loader, num_classes,
                                            args.unlearn_epoch, args.unlearn_rate,
                                            logger = logger, console_handler = console_handler, 
                                            loader_dict=loader_dict, experiment_path = experiment_path, disable_bn = disable_bn,
                                            adv_eps = args.adv_eps, adv_lambda=args.adv_lambda, 
                                            reg_lambda=args.reg_lambda) # 使用相同的函数，只是添加了一个参数
    elif args.method == "fisher":
        unlearn_model = method.fisher(ori_model, train_forget_loader, train_remain_loader,
                                                    alpha=args.alpha, num_classes=num_classes,
                                                    logger=logger, console_handler=console_handler,
                                                    loader_dict=loader_dict, experiment_path = experiment_path,
                                                    freeze_linear = args.freeze_linear)
    elif args.method == "wood_fisher":
        train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 45000
        train_remain_loader_sole = torch.utils.data.DataLoader(dataset=trainset, batch_size=1,    # bs设置为1用于专门的wfisher计算
                                                            sampler=train_remain_sampler,
                                                            num_workers=num_workers)

        unlearn_model = method.wood_fisher(ori_model, train_forget_loader, train_remain_loader, train_remain_loader_sole, 
                                                    alpha=args.alpha,
                                                    retain_data=args.retain_data,
                                                    logger=logger, console_handler=console_handler,
                                                    loader_dict=loader_dict, experiment_path= experiment_path)
    elif args.method == 'delete':
        unlearn_model = method.delete(ori_model, train_forget_loader,
                                                    args.unlearn_epoch, args.unlearn_rate,
                                                    logger=logger, console_handler=console_handler,
                                                    loader_dict=loader_dict, experiment_path= experiment_path, disable_bn = disable_bn,
                                                    ############## 我的额外自定义参数开始
                                                    soft_label=args.soft_label
        )
    elif args.method == 'ablation': 
        unlearn_model = method.my_method_ablation(ori_model, train_forget_loader,
                                                    args.unlearn_epoch, args.unlearn_rate,
                                                    logger=logger, console_handler=console_handler,
                                                    loader_dict=loader_dict, experiment_path= experiment_path, 
                                                    ############## 我的额外自定义参数开始
                                                    soft_label=args.soft_label,
                                                    alpha = args.ablation_a,
                                                    temperature = args.ablation_t
        )
    elif args.method == 'pass':
        pass
    else:
        raise ValueError('method not found')    # 未找到方法

    if unlearn_model:
        # torch.save(unlearn_model.state_dict(), path / description / vice_description / f"ckpt.pth")
        torch.save(unlearn_model, path / description / method_description / vice_description / f"ckpt.pth")
        # 快速开发直接保存模型，不保存参数。但是在不同的开发环境下运行可能导致错误。
        # 因为lora的原因，暂时直接保存模型，不保存参数

        # NOTE:默认在unlearn method已经打印过了
        # note_print(f"\nunlearn model的性能是")
        # now = time.time()
        # _, test_acc         = test(unlearn_model, test_loader)
        # _, forget_acc       = test(unlearn_model, test_forget_loader)
        # _, remain_acc       = test(unlearn_model, test_remain_loader)
        # _, train_forget_acc = test(unlearn_model, train_forget_loader)
        # _, train_remain_acc = test(unlearn_model, train_remain_loader)

        # logger.info('test acc:{:.2%}, train forget acc:{:.2%}, train remain acc:{:.2%}, test forget acc:{:.2%}, test remain acc:{:.2%}\n taken time {}'
        #     .format(test_acc, train_forget_acc, train_remain_acc, forget_acc, remain_acc, time.time()-now)) # 好像是不需要，注意一下
    # print('\nretrain model acc:\n', test_each_classes(unlearn_model, test_loader, num_classes))
        
    import evaluation
    test_remain_len    = len(test_remain_index)

    # 从train_remain_index中随机选取test_len个
    import random
    random.shuffle(train_remain_index)

    train_remain_index = train_remain_index[:test_remain_len]
    logger.info(f"train remain size: {len(train_remain_index)}")
    train_remain_sampler = SubsetRandomSampler(train_remain_index)  # 重新采一个train remain，大小和test remain相同
    train_remain_loader = DataLoader(train_remain_loader.dataset, batch_size=args.batch_size, sampler=train_remain_sampler)

    if args.train_from_scratch or args.method == "pass":
        mia_result = evaluation.SVC_MIA(
            shadow_train=train_remain_loader,
            shadow_test=test_remain_loader,
            target_train=train_forget_loader,
            target_test=None,
            model=ori_model,
        )
        print(f"original model\n {mia_result}")
    if args.retrain_from_scratch or args.method == "pass":
        mia_result = evaluation.SVC_MIA(
            shadow_train=train_remain_loader,
            shadow_test=test_remain_loader,
            target_train=train_forget_loader,
            target_test=None,
            model=retrain_model,
        )
        print(f"retrain model\n {mia_result}")
    if unlearn_model:
        logger.info("start mia evaluation")
        mia_result = evaluation.SVC_MIA(
            shadow_train=train_remain_loader,
            shadow_test=test_remain_loader,
            target_train=train_forget_loader,
            target_test=None,
            model=unlearn_model,
        )
        logger.info(f"unlearn model\n {mia_result}")
    logger.info("运行结束")
    exit()
