import torch
import torch.nn as nn
import torch.nn.functional as F
# import kornia
import ipdb

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class L2LossWithIgnore(nn.Module):

    def __init__(self, ignore_value=None):
        super(L2LossWithIgnore, self).__init__()
        self.ignore_value = ignore_value

    def forward(self, input, target): # N1HW, N1HW
        if self.ignore_value is not None:
            target_area = target != self.ignore_value
            target = target.float()
            return (input[target_area] - target[target_area]).pow(2).mean()
        else:
            return (input - target.float()).pow(2).mean()



class MaskWeightedCrossEntropyGaussian(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super().__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight

    def forward(self, predict, target, mask):
        '''
        predict: NCHW
        target: NHW
        mask: NHW
        '''
        n, c, h, w = predict.size()
        mask = mask.bool()
        target = target.float()

        mean, std = predict[:, 0], predict[:, 1]
        mean = mean.sigmoid()
        std = F.softplus(std) + 1e-6

        pre_loss = 1/2 * (F.binary_cross_entropy(mean, target, reduction='none')/std.pow(2)) + 1/2 * std.pow(2)

        loss = (self.inmask_weight * pre_loss[mask==1].sum() + self.outmask_weight * pre_loss[mask==0].sum()) / (n * h * w+1)

        return loss


class MaskWeightedNLLGaussian(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super().__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight

    def forward(self, predict, target, mask):
        '''
        predict: NCHW
        target: NHW
        mask: NHW
        '''
        n, c, h, w = predict.size()
        # ipdb.set_trace()
        mask = mask.bool()
        target = target.float()

        mean, std = predict[:, 0], predict[:, 1]
        mean = mean.sigmoid()
        std = F.softplus(std) + 1e-6

        pre_loss = 1/2 * ((target - mean)/std).pow(2) + 1/2 * std.pow(2)
        # ipdb.set_trace()

        loss = (self.inmask_weight * pre_loss[mask==1].sum() + self.outmask_weight * pre_loss[mask==0].sum()) / (n * h * w+1)
        return loss



class MaskWeightedNLLExponential(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super().__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight

    def forward(self, predict, target, mask):
        '''
        predict: NCHW
        target: NHW
        mask: NHW
        '''
        n, c, h, w = predict.size()
        mask = mask.bool()
        target = target.float()

        mean = predict[:, 0]
        mean = mean.sigmoid().clamp(min=1e-16) # divide the mean error

        pre_loss = torch.where(target == 1, 1/mean + 1/2*mean.pow(2) - 3/2,  1/2*mean.pow(2))

        loss = (self.inmask_weight * pre_loss[mask==1].sum() + self.outmask_weight * pre_loss[mask==0].sum()) / (n * h * w)
        return loss



class MaskWeightedNLLGaussianBoundary2(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super().__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight

    def forward(self, predict, target, mask):
        '''
        predict: NCHW
        target: NHW
        mask: NHW
        '''
        n, c, h, w = predict.size()
        mask = mask.bool()

        std_target = target[:, 1].float()
        target = target[:, 0].float()

        mean, std = predict[:, 0], predict[:, 1]
        mean = mean.sigmoid()
        std = std.sigmoid().clamp(min=1e-16)

        pre_loss = 1/2 * ((target - mean)/std).pow(2)
        # pre_loss = 1/2 * ((target - mean)/std).pow(2) + 1/2 * std.pow(2)

        pre_loss = (self.inmask_weight * pre_loss[mask==1].sum() + self.outmask_weight * pre_loss[mask==0].sum()) 

        std_loss = F.binary_cross_entropy(std, std_target, reduction='none')

        std_loss = (self.inmask_weight * 2 * std_loss[std_target==1].sum() + self.outmask_weight * std_loss[std_target==0].sum())

        loss = (pre_loss + std_loss) / (n * h * w)

        return loss


class MaskWeightedNLLGaussianBoundary(nn.Module):

    def __init__(self, inmask_weight=5, outmask_weight=1):
        super().__init__()
        self.inmask_weight = inmask_weight
        self.outmask_weight = outmask_weight
        print('go MaskWeightedNLLGaussianBoundary MSE')

    def forward(self, predict, target, mask):
        '''
        predict: NCHW
        target: NHW
        mask: NHW
        '''
        n, c, h, w = predict.size()
        mask = mask.bool()

        std_target = target[:, 1].float()
        target = target[:, 0].float()

        mean, std = predict[:, 0], predict[:, 1]
        mean = mean.sigmoid()
        
        std = F.softplus(std) + 1e-16

        pre_loss = 1/2 * ((target - mean)/std).pow(2) # better performance
        # pre_loss = 1/2 * ((target - mean)/std).pow(2) + 1/2 * std.pow(2) # this is new

        pre_loss = (self.inmask_weight * pre_loss[mask==1].sum() + self.outmask_weight * pre_loss[mask==0].sum()) 

        std_loss = F.mse_loss(std, std_target, reduction='none')

        std_loss = (self.inmask_weight * std_loss[std_target==1].sum() + self.outmask_weight * std_loss[std_target==0].sum())

        loss = (pre_loss + std_loss) / (n * h * w)

        return loss




def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, input, mask, output, gt):
        loss_dict = {}
        output_comp = mask * input + (1 - mask) * output

        loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['prc'] = 0.0
        for i in range(3):
            loss_dict['prc'] += self.l1(feat_output[i], feat_gt[i])
            loss_dict['prc'] += self.l1(feat_output_comp[i], feat_gt[i])

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            loss_dict['style'] += self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))

        loss_dict['tv'] = total_variation_loss(output_comp)

        return loss_dict
