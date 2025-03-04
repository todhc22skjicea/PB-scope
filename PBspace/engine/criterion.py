
from utils.misc import export_fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import math

EPS = 10 ** -6

class CCLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(CCLoss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        self.temperature = temperature

        self.bs = None
        self.batch_indexes = None
        self.positive_mask = None
        self.negative_mask = None
        self.labels = None

        self.cluster_labels = None
        self.cluster_negative_mask = None
        self.cluster_positive_mask = None
        self.cluster_indexes = None

    def instance_loss(self, z1, z2):
        bs = z1.shape[0]
        if self.batch_indexes is None or self.bs != bs:
            self.bs = bs
            self.batch_indexes = torch.arange(bs, device=z1.device)
            self.positive_mask = F.one_hot(torch.cat([self.batch_indexes + bs, self.batch_indexes]), bs * 2).bool()
            self.negative_mask = (1 - F.one_hot(torch.cat([self.batch_indexes, self.batch_indexes + bs]), bs * 2)-self.positive_mask.float()).bool()
            self.labels = torch.zeros((2*bs,),device=z1.device).long()

        z = F.normalize(torch.cat([z1, z2], dim=0),p=2,dim=-1)
        s = z @ z.T
        positives = s[self.positive_mask].view(2 * bs, 1)
        negatives = s[self.negative_mask].view(2 * bs, -1)
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        loss = self.CE(logits, self.labels)
        return loss

    def cluster_loss(self, p1, p2):
        if len(p1.shape)==2:
            p1 = p1.unsqueeze(0)
        if len(p2.shape)==2:
            p2 = p2.unsqueeze(0)
        k,bs,c = p1.shape
        if self.cluster_indexes is None or self.bs != bs:
            self.bs = bs
            self.cluster_indexes = torch.arange(c, device=p1.device)
            self.cluster_positive_mask = torch.stack(k*[F.one_hot(torch.cat([self.cluster_indexes + c, self.cluster_indexes]), c * 2).bool()])
            negative_mask = torch.stack(k*[F.one_hot(torch.cat([self.cluster_indexes, self.cluster_indexes + c]), c * 2)])
            self.cluster_negative_mask = (1 - negative_mask - self.cluster_positive_mask.float()).bool()
            self.cluster_labels = torch.zeros((2*c,),device=p1.device).long()

        p = torch.cat([p1, p2], dim=2)
        if len(p.shape)==2:
            p = p.unsqueeze(0)
        p = F.normalize(p,dim=1)
        s = torch.einsum("kna, knb->kab", p, p)
        positives = s[self.cluster_positive_mask].view(k,2*c,-1)
        negatives = s[self.cluster_negative_mask].view(k,2*c,-1)
        logits = torch.cat([positives, negatives], dim=2)

        loss_ce = []
        loss_ne = []
        for k_ in range(k):
            loss_ce.append(self.CE(logits[k_], self.cluster_labels))
            p_i = p1[k_].sum(0).view(-1)
            p_i /= p_i.sum()
            ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
            p_j = p2[k_].sum(0).view(-1)
            p_j /= p_j.sum()
            ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
            loss_ne.append(ne_i + ne_j)
        return loss_ce, loss_ne

    def forward(self, p1, p2, z1,z2):
        loss_ce, loss_ne = self.cluster_loss(p1,p2)
        loss_cc = self.instance_loss(z1, z2)
        return loss_ce, loss_ne, loss_cc



def clustering_accuracy_metrics(cluster_labels, ground_truth):
    if isinstance(cluster_labels, torch.Tensor):
        cluster_labels = cluster_labels.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if len(cluster_labels.shape) == 1:
        cluster_labels = np.expand_dims(cluster_labels, 0)

    cluster_labels = cluster_labels.astype(np.int64)
    ground_truth = ground_truth.astype(np.int64)
    assert cluster_labels.shape[-1] == ground_truth.shape[-1]

    metrics = {}
    cluster_accuracies, cluster_nmis, cluster_aris = [], [], []
    interclustering_nmi = []
    clusterings = len(cluster_labels)
    for k in range(clusterings):
        for j in range(clusterings):
            if j>k:
                interclustering_nmi.append(np.round(normalized_mutual_info_score(cluster_labels[k], cluster_labels[j]), 5))
        cluster_accuracies.append(clustering_acc(cluster_labels[k], ground_truth))
        cluster_nmis.append(np.round(normalized_mutual_info_score(cluster_labels[k], ground_truth), 5))
        cluster_aris.append(np.round(adjusted_rand_score(ground_truth, cluster_labels[k]), 5))
        metrics["cluster_acc_" + str(k)] = cluster_accuracies[-1]
        metrics["cluster_nmi_" + str(k)] = cluster_nmis[-1]
        metrics["cluster_ari_" + str(k)] = cluster_aris[-1]
    '''
    metrics["max_cluster_acc"], metrics["mean_cluster_acc"], metrics["min_cluster_acc"] = np.max(
        cluster_accuracies), np.mean(cluster_accuracies), np.min(cluster_accuracies)
    metrics["max_cluster_nmi"], metrics["mean_cluster_nmi"], metrics["min_cluster_nmi"] = np.max(
        cluster_nmis), np.mean(cluster_nmis), np.min(cluster_nmis)
    metrics["max_cluster_ari"], metrics["mean_cluster_ari"], metrics["min_cluster_ari"] = np.max(
        cluster_aris), np.mean(cluster_aris), np.min(cluster_aris)
    if clusterings>1:
        metrics["interclustering_nmi"] = sum(interclustering_nmi)/len(interclustering_nmi)
    '''
    metrics["mean_cluster_acc"] = np.mean(cluster_accuracies)
    metrics["mean_cluster_nmi"] = p.mean(cluster_nmis)
    metrics["mean_cluster_ari"] = np.mean(cluster_aris)
    if clusterings>1:
        metrics["interclustering_nmi"] = sum(interclustering_nmi)/len(interclustering_nmi)
    return metrics

def clustering_acc(y_pred, y_true):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(*ind)]) * 100.0 / y_pred.size

