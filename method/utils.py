import matplotlib.pyplot as plt
from trainer import test

keys = ["train_forget", "train_remain", "test_forget", "test_remain"]

# eval_opt = {
#     "train_forget": True,
#     "test_forget": True,
#     "train_remain": True,
#     "test_remain": True,
# }
eval_opt = {
    "train_forget": True,
    "train_remain": True,
    "test_forget": True,
    "test_remain": True,
}

def evaluate_model_on_all_loaders(model, loader_dict, eval_option, logger, extra_class=0):
    keys = ["train_forget", "train_remain", "test_forget", "test_remain"]
    current_accs_dict = {key: float("nan") for key in keys}
    
    for key in keys:
        if eval_option[key]:
            _, acc = test(model, loader_dict[key], extra_class)
            current_accs_dict[key] = acc

    logger.info(f"train forget acc:{current_accs_dict['train_forget']:.2%}, train_remain_acc:{current_accs_dict['train_remain']:.2%}, test forget acc:{current_accs_dict['test_forget']:.2%}, test remain acc:{current_accs_dict['test_remain']:.2%}")
    return current_accs_dict

def plot_unlearn_remain_acc_figure(epoch, accs_dict, experiment_path, plot_type="plot"):    
    # 如果只有一个epoch，则折线图无法正确显示。使用scatter
    assert plot_type in ["plot", "scatter"], f"Unknown plot type: {plot_type}"
    plot_method = getattr(plt, plot_type)

    epochs = list(range(1, epoch + 1))
    plt.figure(figsize=(12, 6))

    # 左侧 Forget 部分的图表
    plt.subplot(1, 2, 1)
    
    if not float("nan") in accs_dict["train_forget"]:
        plot_method(epochs, accs_dict["train_forget"], label='Train Forget Accuracy')   # 只能使用plot和scatter都支持的选项
    if not float("nan") in accs_dict["test_forget"]:
        plot_method(epochs, accs_dict["test_forget"], label='Test Forget Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # 固定纵轴范围为 [0, 1]
    plt.title('Forget Accuracy during Unlearning Process')
    plt.legend()
    plt.grid(True)

    # 右侧 Remain 部分的图表
    plt.subplot(1, 2, 2)
    if not float("nan") in accs_dict["train_remain"]:
        plot_method(epochs, accs_dict["train_remain"], label='Train Remain Accuracy')
    if not float("nan") in accs_dict["test_remain"]:
        plot_method(epochs, accs_dict["test_remain"], label='Test Remain Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # 固定纵轴范围为 [0, 1]
    plt.title('Remain Accuracy during Unlearning Process')
    plt.legend()
    plt.grid(True)

    # 调整子图间距
    plt.tight_layout()

    # 保存图像
    plt.savefig(experiment_path / 'unlearn_acc_forget_remain.png')
    plt.show()
    plt.close()