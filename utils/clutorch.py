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
    def __init__(self, dim=256, k_class=12, div=4, temp=0.8):
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

    def forward(self, x1, x2, x3):
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
        # y = F.one_hot(y)
        # loss_sup = -0.5 * torch.mean(y * torch.log(p_b) + y * torch.log(p_d) + y * torch.log(p_e))

        # swap prediction problem
        loss_un_sup = -0.5 * torch.mean(q_d * torch.log(p_b) + q_e * torch.log(p_d) + q_b * torch.log(p_e))

        # total loss
        # loss = (0.5 * loss_sup + loss_un_sup) / 2.

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

"""
    对比聚类损失 Contrast Clustering Loss
    
    亮点:
    在本文中, 深度学习的特征提取过程被划分为四个阶段: 浅层共有模态特征提取阶段, 深层私有模态特征提取阶段, 语义信息提取阶段和语义信息融合阶段. 具体来
    说, 在浅层共有模态特征提取阶段, 主要利用共享的骨干网络提取多模态超声数据共有的背景信息和环境信息. 随后, 利用提出的全局-局部双分支特征提取模块提
    取不同模态特有的全局特征(形状和色彩信息)和局部特征(细节和纹理信息). 此外, 在语义信息提取阶段, 利用提出的对比聚类损失使各模态图像的语义信息对诊断
    结果具有相同且正确的观点. 同时, 利用可逆神经网络完成对缺失模态的转换, 使网络能够在模态缺失的状态下完成网络的推理过程. 最后, 通过提出的可信动态特
    征融合模态实现不同语义信息的动态融合, 使网络能够充分利用不同模态的语义信息, 进一步增强多模态网络的诊断性能.
    
    本文提出了一种基于聚类思想的对比聚类损失用于对不同模态的语义信息进行一致性约束. 所提约束可以解释为: 同一样本下, 不同模态的超声图像的语义特征通过
    聚类中心后应隶属于同一类别; 同一模态下, 真实类别相同的不同超声样本对应的语义特征在聚类后的类别相同, 且真实类别不同的超声样本对应的语义特征在聚类
    后的类别不同. (聚类分配, 聚类原型)
    
    对比聚类损失由监督损失和无监督损失两部分组成, 其中监督损失的计算公式如下:
    loss_sup = -0.5 * torch.mean(y * torch.log(p_b) + y * torch.log(p_d) + y * torch.log(p_e))
    其中, p 为利用聚类中心和softmax函数生成的概率. 该部分损失函数通过最小化聚类结果和真实标签之间的差异, 使网络能够在标签信号的监督下对不同模态的语
    义特征进行正确聚类. 即, 同一模态下, 真实类别相同的样本对应的语义特征在聚类后的类别相同, 且真实类别不同的样本对应的语义特征在聚类后的类别不同. 相
    应的, 无监督损失的计算公式如下:
    loss_un_sup = -0.5 * torch.mean(q_d * torch.log(p_b) + q_e * torch.log(p_d) + q_b * torch.log(p_e))
    其中, q 为利用 Sinkhorn-Knopp 算法得到的无监督聚类结果. 具体来说, 该算法将原型分数转换为概率分布, 而分布上的每个数据点则表示各模态特征被分配
    到各原型的概率. 因此, 该部分损失函数通过交换预测的方法最小化不同模态图像的聚类分配结果之间的差异, 使网络能够保证同一样本下不同模态之间的聚类分配
    的一致. 即: 同一样本下, 不同模态的超声图像的语义特征通过聚类中心后应隶属于同一类别. 通过最小化不同模态图像之间的分布差异, 有助于提高多模态网络诊
    断结果的一致, 从而有效提高网络的诊断性能. 合并后的损失函数如下:
    loss = loss_sup + loss_un_sup
    通过上述两部分损失函数的共同作用, 网络能够在保持诊断结果一致性的同时, 对超声图像进行准确诊断. 
    
    其中, 矩阵 C 为可训练的聚类原型, 计算输入特征与聚类中心的内积得到描述原始特征信息的原型分数 score. 随后, 利用 Sinkhorn-Knopp 算法在最大化
    原型分数和聚类分配之间相似度的同时, 将原型分数转换为无监督聚类的分配结果. 
    max Tr(QCZ)+H(Q) 
    同时, 为了防止模型将所有样本映射到同一聚类中心从而导致模型坍塌的问题, 本文对聚类条件进行了熵正则化条件约束, 从而使同一批次内的不同样本能够相对均
    匀的聚类到不同类别中. 约束条件如下:
    
    此外, 由于离散化聚类分配是比梯度更新更激进的优化步骤, 使得网络快速收敛的同时也会导致较差的性能. 因此, 本文保留了软编码的聚类分配Q*.
    
    什么是"平凡解" ?
    在最优传输问题中, 我们的目标是找到一个传输方案, 使得总的传输成本最小. 然而, 如果没有任何约束, 那么所有的物品都可能被分配到成本最低的地方, 这就是
    所谓的"平凡解".
    在无监督损失函数中, 其目标是使同一样本下各模态的聚类结果一致. 然而, 如果没有任何约束, 那么所有样本的聚类结果都可能会被分配为同一类别, 从而导致模
    型坍塌. 

"""
