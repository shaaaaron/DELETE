import torch.nn as nn
import torch

def _find_z(model, inputs, targets, h):
    '''
    Finding the direction in the regularizer
    '''
    inputs.requires_grad_()
    outputs = model(inputs)
    loss_z = nn.CrossEntropyLoss()(model(inputs), targets)
    # loss_z.backward(torch.ones(targets.size()).to(self.device))
    loss_z.backward()
    grad = inputs.grad.data + 0.0
    norm_grad = grad.norm().item()
    z = torch.sign(grad).detach() + 0.  ###[64, 3, 32, 32]
    z = 1. * (h) * (z + 1e-7) / (z.reshape(z.size(0), -1).norm(dim=1)[:, None, None, None] + 1e-7)  ###[64, 3, 32, 32]
    # zero_gradients(inputs)
    inputs.grad.zero_()
    model.zero_grad()

    return z, norm_grad


def curvature(model, inputs, targets, h=3., lambda_=4):
    '''
    Regularizer term in CURE
    '''
    z, norm_grad = _find_z(model, inputs, targets, h)

    inputs.requires_grad_()
    outputs_pos = model(inputs + z)
    outputs_orig = model(inputs)

    loss_pos = nn.CrossEntropyLoss()(outputs_pos, targets)
    loss_orig = nn.CrossEntropyLoss()(outputs_orig, targets)
    grad_diff = torch.autograd.grad((loss_pos - loss_orig), inputs, create_graph=True)[0]
    ##grad_outputs=torch.ones(targets.size()).to(self.device),
    # curv_profile = torch.sort(grad_diff.reshape(grad_diff.size(0), -1))[0]  ###[64, 3072]
    reg = grad_diff.reshape(grad_diff.size(0), -1).norm(dim=1)  ###[64]
    # del grad_diff
    model.zero_grad()

    return torch.sum(lambda_ * reg) / float(inputs.size(0)), reg


def PM(logit, target):#[128,10], [128]
    if logit.shape[1] == 10:
        eye = torch.eye(10).cuda() #[10, 10]
    else:
        eye = torch.eye(11).cuda()
    # tmp1 = eye[target]#转one-hot
    # tmp2 = logit.softmax(1)#【128，10】
    # tmp3 = tmp1*tmp2
    # tmp3 = tmp3.sum(1)
    probs_GT = (logit.softmax(1) * eye[target]).sum(1).detach()#[128]
    top2_probs = logit.softmax(1).topk(2, largest = True)#[128, 2]
    # tmp4 = (top2_probs[1] == target.view(-1,1)).float()#[128, 2]
    # tmp4 = tmp4.sum(1)#[128]
    # tmp4 = tmp4 == 1#[128]bool
    GT_in_top2_ind = (top2_probs[1] == target.view(-1,1)).float().sum(1) == 1#[128]bool
    probs_2nd = torch.where(GT_in_top2_ind, top2_probs[0].sum(1) - probs_GT, top2_probs[0][:,0]).detach()
    return  probs_2nd - probs_GT


def weight_assign(logit, target, bias, slope):
    pm = PM(logit, target)
    reweight = ((pm + bias) * slope).sigmoid().detach()
    normalized_reweight = reweight * 3
    return normalized_reweight