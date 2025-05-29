import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import *
from models import *
import tqdm


# def loss_picker(loss):
#     if loss == 'mse':
#         criterion = nn.MSELoss()
#     elif loss == 'cross':
#         criterion = nn.CrossEntropyLoss()   # 默认计算batch上的mean
#         # cross中有计算logsoftmax和nll loss的两部分，模型最后输出的logits不要经过softmax
#     else:
#         raise ValueError("loss function not found")

#     return criterion


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError("loss function not found")
    return optimizer


def train(model, data_loader, optimizer, epoch, tqdm_on=True):    # epoch只是为了显示进度条
    model.train()
    # running_loss = 0.0
    # correct = 0
    # total = 0
    criterion = nn.CrossEntropyLoss()

    # for step, (batch_x, batch_y) in enumerate(tqdm.tqdm(data_loader)):
    if tqdm_on:
        for inputs, labels in tqdm.tqdm(data_loader, desc=f"Epoch {epoch}"):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")    # 64, 3, 32, 32
            optimizer.zero_grad()

            outputs = model(inputs)  # 64, 10
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    else:
        for inputs, labels in data_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()

            outputs = model(inputs)  # 64, 10
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # running_loss += loss.item()
        # _, predicted = outputs.max(1)
        # total += labels.size(0)
        # correct += predicted.eq(labels).sum().item()
        # 在for循环过程中同时计算acc/loss和更新可能导致不准确


def test(model, data_loader, extra_class=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in (data_loader):
        # for inputs, labels in tqdm.tqdm(data_loader):
            inputs, labels = inputs.to("cuda"), labels.to("cuda")

            outputs = model(inputs)
            if extra_class!=0:
                outputs = outputs[:, :-extra_class]
                
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(data_loader)
    train_acc = correct / total    # 是浮点数，在打印的时候通过.2%直接转化
    return train_loss, train_acc


@timer
def train_save_model(train_loader, test_loader, model_name, optim_name, learning_rate, num_epochs, path, description):
    # NOTE:部分数据集可能因为targets和labels出错，也可能因为遗忘superclass导致不足标准num_classes
    num_classes = max(train_loader.dataset.targets) + 1  # if args.num_classes is None else args.num_classes
    print(f"num_classes:{num_classes}")

    model = get_model(model_name, num_classes)
    model = model.to("cuda")
    print(f"Model {model_name} loaded")

    if torch.cuda.device_count() > 1:   # INFO: 代码运行的时候总是单GPU提供CUDA_VISIBLE_DEVICES
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate)
    # adam没有momentum参数，但是有beta参数。代码中没有设置使用默认
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)  # NOTE: 原代码没有使用scheduler，但我觉得影响不大
    best_acc = 0

    # 保存训练和测试过程中的损失和准确率
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    def format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes)}:{seconds:.0f}"
    
    start_time = time.time()
    for epoch in range(num_epochs):
        train(model=model, data_loader=train_loader, optimizer=optimizer, epoch=epoch)

        train_loss, train_acc = test(model=model, data_loader=train_loader)
        print(f"Train Loss: {train_loss:.2f}, Train Accuracy: {train_acc:.2%}")
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss,  test_acc = test(model=model, data_loader=test_loader)
        print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2%}")
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), path / f"{description}.pth") # pt和pth没有区别。惯例上pt保存model，pth保存state_dict

        #NOTE: scheduler.step()在每个epoch结束后调用
        scheduler.step()

        # 保存准确率和损失曲线
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1) 
        plt.legend()
        plt.title('Accuracy Curve')

        plt.tight_layout()
        plt.savefig(path/f'{description}_training_curves.png')
        plt.close()
        now_time = time.time()
        token_time = now_time - start_time
        total_time = (now_time-start_time)/(epoch + 1)*num_epochs
        left_time = total_time - token_time
        print(f"Time taken: {format_time(token_time)}/{format_time(total_time)}, Left time: {format_time(left_time)}")
    print(f"Model saved at { path/f'{description}.pth'}")

    return model


@timer
def finetune_save_model(train_loader, test_loader, model, optim_name, learning_rate, num_epochs, path, description):
    # NOTE:专门用于vgg数据集和resnet18模型

    optimizer = optimizer_picker(optim_name, model.parameters(), lr=learning_rate)

    best_acc = 0

    # 保存训练和测试过程中的损失和准确率
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    def format_time(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes)}:{seconds:.0f}"
    
    start_time = time.time()
    for epoch in tqdm.tqdm(range(num_epochs)):
        train(model=model, data_loader=train_loader, optimizer=optimizer, epoch=epoch, tqdm_on=False)

        train_loss, train_acc = test(model=model, data_loader=train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_loss,  test_acc = test(model=model, data_loader=test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), path / f"{description}.pth")

        # 保存准确率和损失曲线
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1) 
        plt.legend()
        plt.title('Accuracy Curve')

        plt.tight_layout()
        plt.savefig(path/f'{description}_training_curves.png')
        plt.close()
        now_time = time.time()
        token_time = now_time - start_time
        total_time = (now_time-start_time)/(epoch + 1)*num_epochs
        left_time = total_time - token_time
        print(f"Time taken: {format_time(token_time)}/{format_time(total_time)}, Left time: {format_time(left_time)}")
    print(f"Model saved at { path/f'{description}.pth'}")

    return model


def test_each_classes(model, loader, num_classes=10):
    model.eval()
    res = ''

    cnt = torch.zeros(num_classes).to(torch.int64).cuda()
    pred_cnt = torch.zeros(num_classes).to(torch.int64).cuda()  # 必须放在int64上，否则后面无法scatter_add
    with torch.no_grad():
        for data, target in loader:
            data = data.cuda()
            target = target.cuda()

            output = model(data)
            probabilities = F.softmax(output, dim=1)    # 获取softmax概率

            # 获取最大概率的预测标签
            pred = probabilities.argmax(dim=1)
            correct = (pred == target).to(torch.int64)  # 将布尔值转换为浮点数，以便进行数值计算

            # 累加每个类别的样本数量和正确预测的数量
            cnt.scatter_add_(0, target, torch.ones_like(target))    # ones like保留了int64
            pred_cnt.scatter_add_(0, target, correct)               # correct已经显式int64转化

    accuracy = pred_cnt / cnt
    for i in range(num_classes):
        res += f'class {i} acc: {accuracy[i]:.2%}\n'
    return res


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu', name=''):
    # NOTE: 实现了混淆矩阵，可能有用
    model.eval()  # switch to eval status

    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        if mode == 'pruned':
            batch_y_predict = batch_y_predict[:, 0:10]

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)   # 将所有结果拼接成一维tensor
    # cat是在已有的维度上进行拼接，stack是在新的维度上进行拼接
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    # print()

    if print_perform and mode != 'backdoor' and mode != 'widen' and mode != 'pruned':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
    if print_perform and mode == 'widen':
        class_name = data_loader.dataset.classes.append('extra class')
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=class_name, digits=4))
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.show()
    if print_perform and mode == 'pruned':  #NOTE: 混淆矩阵，有用吗？
        # print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
        class_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.title('{} confusion matrix'.format(name), loc='center')
        plt.show()

    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc


if __name__ == "__main__":
    class identity_model:
        def __call__(self, x):
            return x
    model = identity_model()
    dataset = []
    i=75
    wrong_data = torch.cat([torch.zeros((100-i, 1)), torch.ones((100-i, 1))], dim=1)
    right_data = torch.cat([torch.ones((i, 1)), torch.zeros((i, 1))], dim=1)
    data = torch.cat([wrong_data, right_data], dim=0)
    label = torch.zeros((100,)).to(torch.int64)
    dataset = dataset + [(data, label)]
    print(test_each_classes(model, dataset, 2))