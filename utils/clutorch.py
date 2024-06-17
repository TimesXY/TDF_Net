import torch
import torch.nn as nn
import torch.nn.functional as F


# def sinkhorn(scores, eps=0.05, n_iter=3):
#     q = torch.exp(-scores / eps).T
#     q = q / torch.sum(q)
#     k, b = q.shape[0], -q.shape[1]
#
#     for _ in range(n_iter):
#         q = q / torch.sum(q, dim=1, keepdim=True)
#         q = q / k
#         q = q / torch.sum(q, dim=0, keepdim=True)
#         q = q / b
#     return (q * b).T


def sk_dist(scores, eps=0.05, n_iter=3):
    q = torch.exp(-scores / eps).T
    q /= q.sum(dim=0)

    k, b = q.shape
    r = torch.ones(k) / k
    c = torch.ones(b) / b

    for _ in range(n_iter):
        q *= (r.cuda() / q.sum(dim=1)).unsqueeze(1)
        q *= (c.cuda() / q.sum(dim=0)).unsqueeze(0)

    return (q / q.sum(dim=0, keepdim=True)).T


class CCLoss(nn.Module):
    def __init__(self, dim=256, k_class=2, div=4, temp=0.8):
        super().__init__()

        # Establishment of clustering centers
        self.temp = temp
        self.prototypes = nn.Parameter(torch.randn(int(dim / div), k_class))

        # Create a global average pooling layer
        self.gap1 = nn.Sequential(nn.Conv2d(dim, int(dim / div), kernel_size=(1, 1)),
                                  nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten())

        self.gap2 = nn.Sequential(nn.Conv2d(dim, int(dim / div), kernel_size=(1, 1)),
                                  nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten())

        self.gap3 = nn.Sequential(nn.Conv2d(dim, int(dim / div), kernel_size=(1, 1)),
                                  nn.ReLU(),
                                  nn.AdaptiveAvgPool2d((1, 1)),
                                  nn.Flatten())

    def forward(self, x1, x2, x3, y):
        # reshape of the tensor
        x1 = self.gap1(x1)
        x2 = self.gap2(x2)
        x3 = self.gap3(x3)

        # normalize prototypes
        # self.prototypes = nn.Parameter(F.normalize(self.prototypes.detach(), dim=1, p=2))

        # prototype scores:
        scores_b = torch.mm(x1, self.prototypes)
        scores_d = torch.mm(x2, self.prototypes)
        scores_e = torch.mm(x3, self.prototypes)

        # compute assignments
        with torch.no_grad():
            q_b = sk_dist(scores_b)
            q_d = sk_dist(scores_d)
            q_e = sk_dist(scores_e)

        # convert scores to probabilities
        p_b = F.softmax(scores_b / self.temp, dim=1)
        p_d = F.softmax(scores_d / self.temp, dim=1)
        p_e = F.softmax(scores_e / self.temp, dim=1)

        # supervised contrastive clustering loss
        y = F.one_hot(y)
        loss_sup = -0.5 * torch.mean(y * torch.log(p_b) + y * torch.log(p_d) + y * torch.log(p_e))

        # swap prediction problem
        loss_un_sup = -0.5 * torch.mean(q_d * torch.log(p_b) + q_e * torch.log(p_d) + q_b * torch.log(p_e))

        # total loss
        loss = (0.5 * loss_sup + loss_un_sup) / 2.

        # normalize prototypes
        with torch.no_grad():
            self.prototypes = nn.Parameter(F.normalize(self.prototypes.detach(), dim=0, p=2))

        return loss_un_sup, self.prototypes


if __name__ == '__main__':
    # [batch_size, channel, height, width]
    A1 = torch.randn([32, 256, 14, 14])
    A2 = torch.randn([32, 256, 14, 14])
    A3 = torch.randn([32, 256, 14, 14])

    encoder = CCLoss(dim=256, k_class=12)
    losses, prototypes = encoder(A1, A2, A3)

    print(losses)
    print(prototypes.shape)
