
import torch.nn.functional as F
import torch

def focal_loss(pred, target, alpha=.25, gamma=2) : 
    # inputs = F.sigmoid(inputs)       
    # inputs = inputs.view(-1)
    # targets = targets.view(-1)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss

    focal_loss = torch.mean(focal_loss)
    # elif self.reduction == 'sum':
    #    focal_loss = torch.sum(focal_loss)
    return focal_loss 

def weighted_focal_loss(pred, target, alpha=.25, gamma=2) : 
    # inputs = F.sigmoid(inputs)       
    # inputs = inputs.view(-1)
    # targets = targets.view(-1)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-bce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss


    ## weighted
    focal_loss[:, 20, :, :] *=3 
    focal_loss[:, 21, :, :] *=3 
    focal_loss[:, 25, :, :] *=3 
    focal_loss[:, 26, :, :] *=3 

    focal_loss[:, 19, :, :] *=2 
    focal_loss[:, 22, :, :] *=2 
    focal_loss[:, 23, :, :] *=2 
    focal_loss[:, 24, :, :] *=2 

    focal_loss = torch.mean(focal_loss)
    # elif self.reduction == 'sum':
    #    focal_loss = torch.sum(focal_loss)
    return focal_loss 

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def weighted_dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    loss[:, 20] *= 3
    loss[:, 21] *= 3
    loss[:, 25] *= 3
    loss[:, 26] *= 3 

    loss[:, 19] *= 2
    loss[:, 22] *= 2
    loss[:, 23] *= 2
    loss[:, 24] *= 2

    return loss.mean()

def calc_loss(pred, target, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def focal_dice_loss(pred, target, focal_weight = 0.5):
    focal = focal_loss(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = focal * focal_weight + dice * (1 - focal_weight)
    # exit()
    ## 19~26, 26, 20
    return loss

def weighted_focal_dice_loss(pred, target, focal_weight = 0.5):
    focal = weighted_focal_loss(pred, target)
    # focal_no = focal_loss(pred, target)
    pred = F.sigmoid(pred)
    dice = weighted_dice_loss(pred, target)
    # dice_no = dice_loss(pred, target)
    loss = focal * focal_weight + dice * (1 - focal_weight)
    # loss_no = focal_no * focal_weight + dice_no * (1 - focal_weight)
    ## 19~26, 26, 20
    return loss
        