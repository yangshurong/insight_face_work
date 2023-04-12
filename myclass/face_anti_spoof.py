# -*- coding: utf-8 -*-
# @Time : 20-6-9 上午10:20
# @Author : zhuying
# @Company : Minivision
# @File : anti_spoof_predict.py
# @Software : PyCharm

from torchvision import transforms
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models


class AntiSpoofDetect():
    def __init__(self, path, device='cuda:0'):
        self.transformer = transforms.Compose([
            transforms.Resize([300, 300]),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5070, 0.4764, 0.4551], [
                                 0.2480, 0.2468, 0.2416]),

        ])
        self.device = device
        self.network = Resnet18(False, device)
        self.loss = PatchLoss(
            model_out_feature=self.network.model_out_feature).to(device=device)
        self.load_params(path)

    def load_params(self, path):
        state_dict = torch.load(path)
        self.network.load_state_dict(state_dict['state_dict'])
        self.loss.load_state_dict(state_dict['loss'])

    def forward(self, img):
        
        img1 = self.transformer(img)
        img2 = self.transformer(img)
        feature1 = self.network(img1.unsqueeze(0).to(self.device))
        feature2 = self.network(img2.unsqueeze(0).to(self.device))
        score1 = F.softmax(self.loss.amsm_loss.s *
                           self.loss.amsm_loss.fc(feature1.squeeze(3).squeeze(2)), dim=1)
        score2 = F.softmax(self.loss.amsm_loss.s *
                           self.loss.amsm_loss.fc(feature2.squeeze(3).squeeze(2)), dim=1)

        score = (score1+score2)/2.0
        return score.cpu().detach().numpy()


class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input, target):
        return self.mse(input, target)


class AdMSoftmaxLoss(nn.Module):

    def __init__(self, in_features, out_features, s=30.0, m_l=0.4, m_s=0.1):
        '''
        AM Softmax Loss
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.s = s
        self.m = [m_s, m_l, m_l, m_l, m_l]
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x, labels):
        '''
        input: 
            x shape (N, in_features)
            labels shape (N)
        '''
        labels = torch.argmax(labels, dim=1)
        # assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)

        x = F.normalize(x, dim=1)

        wf = self.fc(x)
        m = torch.tensor([self.m[ele] for ele in labels]).to(x.device)
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0)
                         for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + \
            torch.sum(torch.exp(self.s * excl), dim=1)

        L = numerator - torch.log(denominator)

        return - torch.mean(L)


class PatchLoss(nn.Module):

    def __init__(self, model_out_feature=512, alpha1=1.0, alpha2=1.0):
        super().__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sim_loss = SimilarityLoss()
        self.amsm_loss = AdMSoftmaxLoss(model_out_feature, 5)

    def forward(self, x1, x2, label):
        amsm_loss1 = self.amsm_loss(x1.squeeze(3).squeeze(2), label)
        amsm_loss2 = self.amsm_loss(x2.squeeze(3).squeeze(2), label)
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        sim_loss = self.sim_loss(x1, x2)
        loss = self.alpha1 * sim_loss + self.alpha2 * (amsm_loss1 + amsm_loss2)
        return loss


class Resnet18(nn.Module):

    def __init__(self, pretrained=True, device='cpu'):
        super(Resnet18, self).__init__()
        base_model = models.resnet18(pretrained=pretrained).to(device)
        self.nets = nn.Sequential(*(list(base_model.children())[:-1]))
        self.device = device
        self.model_out_feature = 512

    def forward(self, x):
        return self.nets(x)
