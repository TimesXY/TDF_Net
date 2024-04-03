import torch
import torch.nn as nn
import torch.nn.functional as F

from block import TransformerBlock


def kl_loss(alpha, num_class):
    # kl loss function -- create the vector beta
    beta = torch.ones((1, num_class))

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


class FuseClassifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        """
        The output layer is built and the SoftPlus activation function is applied to the output layer.
        classifier_dims: the dimension of the projection
        classes: the number of the classes
        """
        super(FuseClassifier, self).__init__()
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
    def __init__(self, dim, classes, annealing_epoch=1, depth=1, heads=4, dim_head=64, mlp_dim=256, dropout=0.):
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
        self.cls = FuseClassifier(dim, classes)

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
        x_cls = self.cls(x_fuse)

        # get the output of different modes
        losses = torch.stack((loss0, loss1, loss2), dim=0)
        evidences = torch.stack((evidence0, evidence1, evidence2), dim=0)
        uncertainties = torch.stack((uncertainty0, uncertainty1, uncertainty2), dim=0)

        # get the loss of the uni-modal
        mean_losses = torch.mean(losses)

        return x_cls, mean_losses, evidences, uncertainties


if __name__ == '__main__':
    # [batch_size, [height width], channel] -> [batch_size, [height width], channel]
    A0 = torch.rand([4, 196, 256])
    A1 = torch.rand([4, 196, 256])
    A2 = torch.rand([4, 196, 256])
    B = torch.tensor([0, 0, 1, 1])
    epoch = 1500

    model = TDFM(dim=256, classes=2, annealing_epoch=1, depth=1, heads=4, dim_head=64, mlp_dim=256, dropout=0.)
    predict_y, loss, evidence, uncertainty = model(A0, A1, A2, B, epoch)

    print(loss)
    print(predict_y.shape)
    print(evidence.shape)
    print(uncertainty.shape)
