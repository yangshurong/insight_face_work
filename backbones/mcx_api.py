import torch
from torch import nn
import torchvision
from torchvision import models
import numpy as np
import sys
import torch.nn.functional as F
from .xception import xception
from backbones.class_layer.inceptionnext import inceptionnext_small
# from model.vit import vit_b_16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def pw_cosine_distance(vector):
    normalized_vec = F.normalize(vector)
    res = torch.mm(normalized_vec, normalized_vec.T)
    cos_dist = 1 - res
    return cos_dist


class API_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(API_Net, self).__init__()
        model_name = 'xception'
        # ---------Resnet101---------
        print("my model name", model_name)
        if model_name == 'res101':
            model = models.resnet101(pretrained=True)
            kernel_size = 14
        # ---------Efficientnet---------
        elif model_name == 'effb0':
            model = models.efficientnet_b0(pretrained=True)
            kernel_size = 14
        # ---------Xception---------
        elif model_name == 'xception':
            model = xception()
            kernel_size = 14
        elif model_name == 'inceptionConvnext':
            model = inceptionnext_small(True)
            kernel_size = 14
        else:
            raise ValueError("Unsupported Backbone!")
        
        if model_name!='inceptionConvnext':
            layers = list(model.children())[:-2]
        else:
            layers = list(model.children())[:-1]

        if 'res' in model_name:
            fc_size = model.fc.in_features
        elif 'eff' in model_name:
            fc_size = model.classifier[1].in_features
        elif 'vit' in model_name:
            fc_size = model.hidden_dim
        elif 'xception' in model_name:
            fc_size = 2048
        elif model_name == 'inceptionConvnext':
            fc_size = 768
        else:
            sys.exit('wrong network name baby')

        self.conv = nn.Sequential(*layers)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1)

        self.map1 = nn.Linear(fc_size * 2, 512)
        self.map2 = nn.Linear(512, fc_size)
        self.fc = nn.Linear(fc_size, num_classes)

        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, targets=None, flag='train', dist_type='euclidean'):

        conv_out = self.conv(images)
        pool_out_old = self.avg(conv_out)
        pool_out = pool_out_old.squeeze()

        if flag == 'train':
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(
                pool_out, targets, dist_type)

            features1 = torch.cat(
                [pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat(
                [pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat(
                [intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat(
                [intra_labels[:, 1], inter_labels[:, 1]], dim=0)
            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1

            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            logit1_self = self.fc(self.drop(features1_self))
            logit1_other = self.fc(self.drop(features1_other))
            logit2_self = self.fc(self.drop(features2_self))
            logit2_other = self.fc(self.drop(features2_other))

            features = self.fc(pool_out)

            return logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2, features

        elif flag == 'features':
            intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(
                pool_out, targets, dist_type)

            features1 = torch.cat(
                [pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
            features2 = torch.cat(
                [pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
            labels1 = torch.cat(
                [intra_labels[:, 0], inter_labels[:, 0]], dim=0)
            labels2 = torch.cat(
                [intra_labels[:, 1], inter_labels[:, 1]], dim=0)
            mutual_features = torch.cat([features1, features2], dim=1)
            map1_out = self.map1(mutual_features)
            map2_out = self.drop(map1_out)
            map2_out = self.map2(map2_out)

            gate1 = torch.mul(map2_out, features1)
            gate1 = self.sigmoid(gate1)

            gate2 = torch.mul(map2_out, features2)
            gate2 = self.sigmoid(gate2)

            features1_self = torch.mul(gate1, features1) + features1
            features1_other = torch.mul(gate2, features1) + features1

            features2_self = torch.mul(gate2, features2) + features2
            features2_other = torch.mul(gate1, features2) + features2

            return features1_self, features1_other, features2_self, features2_other, labels1, labels2

        elif flag == 'val':
            return self.fc(pool_out)
        elif flag == 'test':
            return self.fc(pool_out)
        elif flag == 'tsne':
            return pool_out

    def get_pairs(self, embeddings, labels, dist_type):
        # print(f'embedding shape {embeddings.shape}')
        if dist_type == 'euclidean':
            distance_matrix = pdist(embeddings).detach().cpu().numpy()
        elif dist_type == 'cosine':
            distance_matrix = pw_cosine_distance(
                embeddings).detach().cpu().numpy()
        else:
            sys.exit('wrong distance name baby')

        labels = labels.detach().cpu().numpy().reshape(-1, 1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[(lb_eqs == False)*1] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)

        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)

        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]

            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]

        intra_labels = torch.from_numpy(intra_labels).long().to(device)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_labels = torch.from_numpy(inter_labels).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)

        return intra_pairs, inter_pairs, intra_labels, inter_labels
