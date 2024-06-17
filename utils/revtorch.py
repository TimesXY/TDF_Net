import sys
import torch
import random
import torch.nn as nn
import torch.nn.functional as F


class ReversibleBlock(nn.Module):
    """
    Elementary building block for building (partially) reversible architectures.
    Args:
        in_c: the channel of the input tensor
        out_c: the channel of the output tensor
        split_along_dim (integer): dimension along which the tensor is split into the two parts required.
        fix_random_seed (boolean): Use the same random seed for the forward and backward pass if set to true
    """

    def __init__(self, in_c, out_c, split_along_dim=1, fix_random_seed=True):
        super(ReversibleBlock, self).__init__()

        # Assignment of the parameters.
        self.f_block = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1)))

        # self.g_block = nn.Sequential(
        #     nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), padding=(1, 1)),
        #     nn.GELU(),
        #     nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1)))

        self.h_block = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(3, 3), padding=(1, 1)),
            nn.GELU(),
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1)))

        self.split_along_dim = split_along_dim
        self.fix_random_seed = fix_random_seed
        self.random_seeds = {}

    # initialize random seeds
    def _init_seed(self, namespace):
        if self.fix_random_seed:
            self.random_seeds[namespace] = random.randint(0, sys.maxsize)
            self._set_seed(namespace)

    # set random seeds
    def _set_seed(self, namespace):
        if self.fix_random_seed:
            torch.manual_seed(self.random_seeds[namespace])

    # feed forward in reversible neural network modules
    def forward(self, x):
        """
        Performs the forward pass of the reversible block. Does not record any gradients.
        x: Input tensor. Must be splittable along dimension 1.
        return: Output tensor of the same shape as the input tensor
        """
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        with torch.no_grad():
            self._init_seed('f')
            y1 = x1 + self.f_block(x2)

            # self._init_seed('g')
            # g1 = x1 * torch.exp(self.g_block(y2))

            self._init_seed('h')
            y2 = x2 + self.h_block(y1)

        return torch.cat([y1, y2], dim=self.split_along_dim), self

    @staticmethod
    def reverse(self, y):
        """
        Performs the forward pass of the reversible block. Does not record any gradients.
        x: Output tensor. Must be splittable along dimension 1.
        return: Input tensor of the same shape as the input tensor
        """
        y1, y2 = torch.chunk(y, 2, dim=self.split_along_dim)
        with torch.no_grad():
            self._set_seed('h')
            x2 = y2 - self.h_block(y1)

            # self._set_seed('g')
            # x1 = h2 / torch.exp(self.g_block(y2))

            self._set_seed('f')
            x1 = y1 - self.f_block(x2)

        return torch.cat([x1, x2], dim=self.split_along_dim)


class ModalTransition(nn.Module):
    """
    the missing modal recovery module.
    Args:
        in_c: the channel of the input tensor
        out_c: the channel of the output tensor
        split_along_dim (integer): dimension along which the tensor is split into the two parts required.
        fix_random_seed (boolean): Use the same random seed for the forward and backward pass if set to true
    """

    def __init__(self, in_c, out_c, split_along_dim=1, fix_random_seed=True):
        super(ModalTransition, self).__init__()

        # Building Reversible Modules
        in_c = int(in_c / 2.0)
        out_c = int(out_c / 2.0)

        self.rev_1 = ReversibleBlock(in_c=in_c, out_c=out_c, split_along_dim=split_along_dim,
                                     fix_random_seed=fix_random_seed)

        self.rev_2 = ReversibleBlock(in_c=in_c, out_c=out_c, split_along_dim=split_along_dim,
                                     fix_random_seed=fix_random_seed)

        self.rev_3 = ReversibleBlock(in_c=in_c, out_c=out_c, split_along_dim=split_along_dim,
                                     fix_random_seed=fix_random_seed)

    # feed forward in the missing modal recovery module.
    def forward(self, x1, x2, x3):
        rec_12, self_1 = self.rev_1(x1)
        rec_21 = self.rev_1.reverse(self_1, x2)

        rec_23, self_2 = self.rev_2(x2)
        rec_32 = self.rev_2.reverse(self_2, x3)

        rec_31, self_3 = self.rev_3(x3)
        rec_13 = self.rev_3.reverse(self_3, x1)

        # Recovery features of missing modals
        rec_1 = (rec_31 + rec_21) / 2
        rec_2 = (rec_12 + rec_32) / 2
        rec_3 = (rec_23 + rec_13) / 2

        # compute the probability distribution and log-probability distribution
        prob_1 = F.softmax(x1.view(x1.size(0), -1), dim=-1)
        prob_2 = F.softmax(x2.view(x2.size(0), -1), dim=-1)
        prob_3 = F.softmax(x3.view(x3.size(0), -1), dim=-1)

        log_prob_1 = F.log_softmax(rec_1.view(rec_1.size(0), -1), dim=-1)
        log_prob_2 = F.log_softmax(rec_2.view(rec_2.size(0), -1), dim=-1)
        log_prob_3 = F.log_softmax(rec_3.view(rec_3.size(0), -1), dim=-1)

        # log_prob_31 = F.log_softmax(rec_31.view(rec_31.size(0), -1), dim=-1)
        # log_prob_32 = F.log_softmax(rec_32.view(rec_32.size(0), -1), dim=-1)
        #
        # log_prob_12 = F.log_softmax(rec_12.view(rec_12.size(0), -1), dim=-1)
        # log_prob_13 = F.log_softmax(rec_13.view(rec_13.size(0), -1), dim=-1)
        #
        # log_prob_21 = F.log_softmax(rec_21.view(rec_21.size(0), -1), dim=-1)
        # log_prob_23 = F.log_softmax(rec_23.view(rec_23.size(0), -1), dim=-1)

        # compute KL divergence
        kl_1 = F.kl_div(log_prob_1, prob_1, reduction='batchmean')
        kl_2 = F.kl_div(log_prob_2, prob_2, reduction='batchmean')
        kl_3 = F.kl_div(log_prob_3, prob_3, reduction='batchmean')

        # kl_31 = F.kl_div(log_prob_31, prob_1, reduction='batchmean')
        # kl_21 = F.kl_div(log_prob_21, prob_1, reduction='batchmean')
        #
        # kl_12 = F.kl_div(log_prob_12, prob_2, reduction='batchmean')
        # kl_32 = F.kl_div(log_prob_32, prob_2, reduction='batchmean')
        #
        # kl_13 = F.kl_div(log_prob_13, prob_3, reduction='batchmean')
        # kl_23 = F.kl_div(log_prob_23, prob_3, reduction='batchmean')

        loss_kl_c = (kl_1 + kl_2 + kl_3) / 3.
        # loss_kl_s = (kl_31 + kl_21 + kl_12 + kl_32 + kl_13 + kl_23) / 6.
        # loss_kl = (loss_kl_c + loss_kl_s) / 2.

        return loss_kl_c, self

    @staticmethod
    def inference(self, x1, x2, x3):

        # Missing Modal BUS
        if x1 is None and x2 is not None and x3 is not None:
            x_zeros = torch.zeros_like(x2)
            _, self_ = self.rev_1(x_zeros)
            rec_21 = self.rev_1.reverse(self_, x2)
            rec_31, _ = self.rev_3(x3)
            rec_1 = (rec_31 + rec_21) / 2
            return rec_1, x2, x3

        # Missing Modal DUS
        elif x2 is None and x1 is not None and x3 is not None:
            x_zeros = torch.zeros_like(x1)
            _, self_ = self.rev_2(x_zeros)
            rec_32 = self.rev_2.reverse(self_, x3)
            rec_12, _ = self.rev_1(x1)
            rec_2 = (rec_12 + rec_32) / 2
            return x1, rec_2, x3

        # Missing Modal USE
        elif x3 is None and x1 is not None and x2 is not None:
            x_zeros = torch.zeros_like(x2)
            _, self_ = self.rev_3(x_zeros)
            rec_13 = self.rev_3.reverse(self_, x1)
            rec_23, _ = self.rev_2(x2)
            rec_3 = (rec_23 + rec_13) / 2
            return x1, x2, rec_3

        # Missing Modal BUS and DUS
        elif x1 is None and x2 is None and x3 is not None:
            x_zeros = torch.zeros_like(x3)
            _, self_ = self.rev_2(x_zeros)
            rec_2 = self.rev_2.reverse(self_, x3)
            rec_1, _ = self.rev_3(x3)
            return rec_1, rec_2, x3

        # Missing Modal BUS and USE
        elif x1 is None and x3 is None and x2 is not None:
            x_zeros = torch.zeros_like(x2)
            _, self_ = self.rev_1(x_zeros)
            rec_1 = self.rev_1.reverse(self_, x2)
            rec_3, _ = self.rev_2(x2)
            return rec_1, x2, rec_3

        # Missing Modal DUS and USE
        elif x2 is None and x3 is None and x1 is not None:
            x_zeros = torch.zeros_like(x1)
            _, self_ = self.rev_3(x_zeros)
            rec_3 = self.rev_3.reverse(self_, x1)
            rec_2, _ = self.rev_1(x1)
            return x1, rec_2, rec_3

        else:
            return x1, x2, x3


if __name__ == '__main__':
    # [batch_size, channel, height, width] -> [batch_size, channel, height, width]
    A1 = torch.rand([32, 96, 14, 14])
    A2 = torch.rand([32, 96, 14, 14])
    A3 = torch.rand([32, 96, 14, 14])

    mot = ModalTransition(in_c=96, out_c=96, split_along_dim=2, fix_random_seed=True)
    C, self_c = mot(A1, A2, A3)
    print(C)

    E1 = None
    E2 = None
    D1, D2, D3 = mot.inference(self_c, A1, A2, E2)
    print(D1.shape)
    print(D2.shape)
    print(D3.shape)
