import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import TransformerBlock


class Residual(nn.Module):
    """ 残差结构 """

    def __init__(self, input_channels, output_channels, use_conv=False, strides=1):

        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(3, 3), padding=1,
                               stride=(strides, strides))
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=(3, 3), padding=1)

        if use_conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 1), stride=(strides, strides))
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return F.relu(y)


def resnet_block(input_channels, output_channels, num_residuals, first_block=False):
    """ 建立残差模块 """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, output_channels, use_conv=True, strides=2))
        else:
            blk.append(Residual(output_channels, output_channels))
    return blk


def kl_loss(alpha, num_class):
    # kl loss function -- create the vector beta
    beta = torch.ones((1, num_class)).to("cuda")

    # the sum of the Dirichlet distribution
    s_a = torch.sum(alpha, dim=1, keepdim=True)

    # the sum of the beta
    s_b = torch.sum(beta, dim=1, keepdim=True)

    # according to the first term of equation (9) - 1
    log_alpha = torch.lgamma(s_a) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)

    # according to the first term of equation (9) - 2
    log_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(s_b)

    # according to the second term of equation (9) - 3
    dg0 = torch.digamma(s_a)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + log_alpha + log_beta

    return kl


def ace_loss(p, alpha, num_class, global_step, annealing_step=1):
    # the sum of the Dirichlet distribution
    s = torch.sum(alpha, dim=1, keepdim=True)

    # the evidence of the images
    evidences = alpha - 1

    # one-hot encoder
    label = F.one_hot(p, num_classes=num_class)

    # the adjusted cross-entropy loss
    acl = torch.sum(label * (torch.digamma(s) - torch.digamma(alpha)), dim=1, keepdim=True)

    # the number of iterations is less than the annealing threshold, reduce annealing coefficient
    annealing_coefficient = min(1, global_step / annealing_step)

    # according to formula (7): a^ = y + (1 − y) * alpha
    alp = evidences * (1 - label) + 1

    # kl loss
    dkl = annealing_coefficient * kl_loss(alp, num_class)

    return torch.mean((acl + dkl))


def mse_loss(p, alpha, num_class, global_step, annealing_step=1):
    # the sum of the Dirichlet distribution
    s = torch.sum(alpha, dim=1, keepdim=True)

    # the evidence of the images
    evidences = alpha - 1

    # the class probability
    m = alpha / s

    # one-hot encoder
    label = F.one_hot(p, num_classes=num_class)

    # mean square error loss
    clp = torch.sum((label - m) ** 2, dim=1, keepdim=True)

    # the adjusted cross-entropy loss ??
    acl = torch.sum(alpha * (s - alpha) / (s * s * (s + 1)), dim=1, keepdim=True)

    # the number of iterations is less than the annealing threshold, reduce annealing coefficient
    annealing_coefficient = min(1, global_step / annealing_step)

    # according to formula (7): a^ = y + (1 − y) * alpha
    alp = evidences * (1 - label) + 1

    # kl loss
    dkl = annealing_coefficient * kl_loss(alp, num_class)

    return torch.mean(clp + acl + dkl)


class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        """
        The output layer is built and the SoftPlus activation function is applied to the output layer.
        classifier_dims: the dimension of the projection
        classes: the number of the classes
        """
        super(Classifier, self).__init__()
        self.proj = nn.Sequential(nn.Linear(classifier_dims, classifier_dims))
        self.cls = nn.Sequential(nn.Linear(classifier_dims, classes),
                                 nn.Softplus())

    def forward(self, x):
        x = self.proj(x)
        x = x.mean(dim=1)
        x = self.cls(x)
        return x


class ClassHead(nn.Module):
    def __init__(self, classifier_dims, classes):
        """
        The output layer is built and the SoftPlus activation function is applied to the output layer.
        classifier_dims: the dimension of the projection
        classes: the number of the classes
        """
        super(ClassHead, self).__init__()
        self.proj = nn.Sequential(nn.Linear(classifier_dims, classifier_dims))
        self.cls = nn.Sequential(nn.Linear(classifier_dims, classes))

    def forward(self, x):
        x = self.proj(x)
        x = x.mean(dim=1)
        x = self.cls(x)
        return x


class Dirichlet(nn.Module):
    def __init__(self, classes, classifier_dims, annealing_epoch=1):
        """
        The loss and uncertainties in the Dirichlet distribution.
        classes: number of classification categories
        classifier_dims: dimension of the classifier
        annealing_epoch: KL divergence annealing epoch during training
        """
        super(Dirichlet, self).__init__()

        self.classes = classes
        self.annealing_epoch = annealing_epoch
        self.Classifiers = Classifier(classifier_dims, classes)

    def forward(self, x, y, global_step):
        # [1] consider the output of the neural network as evidence
        evidences = self.Classifiers(x)

        # [2] tectonic Dirichlet distribution (alpha = evidence + 1)
        alpha = evidences + 1

        # [3] the loss function
        losses = ace_loss(y, alpha, self.classes, global_step, self.annealing_epoch)
        losses = torch.mean(losses)

        # [4] Calculating subjective uncertainty
        s = torch.sum(alpha, dim=1, keepdim=True)
        uncertainties = self.classes / s

        return evidences, uncertainties, losses


class TDFM(nn.Module):
    def __init__(self, dim, classes, annealing_epoch=50, depth=1, heads=4, dim_head=64,
                 mlp_dim=256, dropout=0.):
        """
        The loss and uncertainties in the Dirichlet distribution.
        classes: number of classification categories
        annealing_epoch: KL divergence annealing epoch during training
        dim: dimension of the input tensor
        depth: number of the block
        heads: number of the heads
        dim_head: the dimension of the heads
        mlp_dim: the dimension of the mlp
        dropout: dropout rate attention
        """
        super(TDFM, self).__init__()

        # building the self attention module
        self.attention0 = TransformerBlock(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                                           dropout=dropout)
        self.attention1 = TransformerBlock(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                                           dropout=dropout)
        self.attention2 = TransformerBlock(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim,
                                           dropout=dropout)

        # building the output layer of the uni-modal network
        self.dirt0 = Dirichlet(classifier_dims=dim, classes=classes, annealing_epoch=annealing_epoch)
        self.dirt1 = Dirichlet(classifier_dims=dim, classes=classes, annealing_epoch=annealing_epoch)
        self.dirt2 = Dirichlet(classifier_dims=dim, classes=classes, annealing_epoch=annealing_epoch)

        # create the feature characterization layers for modal fusion
        self.proj0 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.proj1 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.proj2 = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))

        # Fusion features used to classify
        self.cls = ClassHead(classifier_dims=dim, classes=classes)

    def forward(self, x0, x1, x2, y, global_step):
        # the input features go through the self-attention module
        x0 = self.attention0(x0) + x0
        x1 = self.attention1(x1) + x1
        x2 = self.attention2(x2) + x2

        # the input features for uni-modal training
        evidence0, uncertainty0, loss0 = self.dirt0(x0, y, global_step)
        evidence1, uncertainty1, loss1 = self.dirt1(x1, y, global_step)
        evidence2, uncertainty2, loss2 = self.dirt2(x2, y, global_step)

        # projection representation of different modes
        x0 = self.proj0(x0) + x0
        x1 = self.proj1(x1) + x1
        x2 = self.proj2(x2) + x2

        # Trusted Dynamic Characterization
        x0 = (torch.ones_like(uncertainty0) - uncertainty0).unsqueeze(2) * x0
        x1 = (torch.ones_like(uncertainty1) - uncertainty1).unsqueeze(2) * x1
        x2 = (torch.ones_like(uncertainty2) - uncertainty2).unsqueeze(2) * x2

        # get the fusion features
        x_fuse = x0 + x1 + x2

        # get the final fusion output and loss
        output = self.cls(x_fuse)

        # get the output of different modes
        losses = torch.stack((loss0, loss1, loss2), dim=0)
        # evidences = torch.stack((evidence0, evidence1, evidence2, evidences_cls), dim=0)
        # uncertainties = torch.stack((uncertainty0, uncertainty1, uncertainty2, uncertainty_cls), dim=0)

        # get the loss of the uni-modal
        mean_losses = torch.mean(losses)

        return output, mean_losses


if __name__ == '__main__':
    # [batch_size, [height width], channel]
    A0 = torch.rand([4, 196, 256])
    A1 = torch.rand([4, 196, 256])
    A2 = torch.rand([4, 196, 256])
    B = torch.tensor([0, 0, 1, 1])
    epoch = 1500

    model = TDFM(dim=256, classes=2, annealing_epoch=1, depth=1, heads=4, dim_head=64,
                 mlp_dim=256, dropout=0.)
    loss, evidence = model(A0, A1, A2, B, epoch)

    print(loss)
    print(evidence.shape)

"""
  亮点:
    在多模态超声图像联合诊断乳腺疾病的过程中, 不同病例下各模态对于诊断结果的贡献度通常存在显著差异. 例如, 对于简单病例, 临床医生根据基础的 B型灰阶超
    声图像就能给出准确的诊断结果, 而面对复杂病例时, 临床医生通常需要联合多种模态的超声图像进行诊断. 因此, 多模态诊断算法需要针对不同样本动态评估各模
    态图像对诊断结果的贡献度, 从而实现多模态特征的动态融合. 为了解决该问题, 本文提出了一种数据依赖的可信动态特征融合模块(TDFM)用于动态融合不同模态图
    像的语义特征. 该模块利用主观逻辑算法将单模态诊断网络的输出建模为狄利克雷分布来获得单模态诊断结果的分配概率和不确定性, 并通过各模态诊断结果的不确定
    性动态调整各模态图像特征表征在联合模态特征表征中所占比例, 实现可信的动态特征融合.

    使用狄利克雷分布来描述诊断结果的分类概率和不确定性, 并通过不确定性实现多模态特征表征的自适应融合. 具体来说, 在该模块中, 单模态诊断网络能够在标签
    信息的监督下为每个模态的特征表征分配一个置信度, 并根据学习到的置信度将各模态的特征表征加权融合为一个更为丰富的具备多模态信息的特征表征.
    
    提出的用于多模态特征融合的可信动态特征融合模块如上图所示. 具体来说, 该模块主要由自注意力机制、置信度生成和可信的动态特征融合三个阶段组成. 首先, 
    利用自注意力机制对输入的不同模态特征进行非线性变换, 以进一步增强所提网络的非线性拟合能力. 
                                                       F_s = MSAM(F_i)
    随后, 利用主观逻辑算法将单模态网络的输出结果建模为狄利克雷分布来获得单模态诊断结果的分配概率和不确定性. 具体的对于视图 v 有:
                                                       u + sum(b_k) = 1
    其中, u为不确定性, bk为k类的分配概率. 同时, 利用主观逻辑将证据狄利克雷分布参数联系起来, 狄利克雷分布参数的计算过程如下:
                                                       alpha = evidence + 1
    其中, evidence 为单模态网络的输出结果, 被视为单模态网络诊断的证据. 而分配概率和不确定性的计算公式如下：
                                                       bk = (alpha - 1) / S 
                                                       u = k / S
                                                       S = sum(alpha)
    其中, S为狄利克雷分布参数的和. 获得不同模态的输出概率和不确定性后, 利用不确定性对不同模态的特征表征进行动态加权和融合. 其计算过程如下所示:
                                                       F_fuse = sum(F_i * u)
    此外, 由于网络输出被作为用于构建狄利克雷分布参数的证据, 导致传统的交叉熵损失不再适用于网络的优化过程. 因此, 通常采用改进后的加权交叉熵损失对网络
    的训练过程进行优化. 改进后的损失函数如下所示:
    上述损失能够促进模型每个样本的正确标签比其他类生成更多的证据, 但是不能保证错误类的证据尽量少. 因此, KL 损失函数被引入用来对证据进行正则化, 以期
    望对于错误分类的样本的证据变为0.
    
    狄利克雷分布是一种多元分布，表示多个选项的概率分布，其参数是一个一维向量，能够表示每个类别或选项的权重，通常用于建模多个分类变量之间的关联关系。它
    的参数可以表示各个分类变量在整体分布中的权重或概率分布。
    
    目前, 基于多模态信息的深度学习技术已经被广泛应用于智能医疗领域, 多模态数据通常能够为计算机辅助诊断系统提供更加丰富的信息, 从而进一步提升诊断系统
    的性能. 然而, 传统的多模态诊断算法通常假设各模态对于诊断结果的贡献是相同的. 但实际上, 在不同样本下, 不同模态对于诊断结果的贡献通常具有动态性,且
    不同患者的同一模态图像所提供信息的重要性也会有所不同. 因此, 本文提出一种可信动态特征融合模块利用单模态网络诊断结果的不确定性对多模态特征进行动态融
    合, 从而提高分类的稳定性和可信性.

"""
