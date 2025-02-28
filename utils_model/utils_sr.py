import os
import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics


# DropPath 实现
def drop_path(x, drop_prob=0., training=False):
    # 若删除概率为 0 或非训练状态，直接返回结果
    if drop_prob == 0. or not training:
        return x

    # 计算保留子路径的概率
    keep_prob = 1 - drop_prob

    # 对不同的二维数据进行处理
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    # 生成随机数组并二值化
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()

    # 标准化后与随机二值化相乘
    output = x.div(keep_prob) * random_tensor

    return output


# DropPath 实现
class DropPath(nn.Module):
    """
    “每个样本”的下降路径（随机深度）（当应用于剩余块的主要路径时）。
    DropPath 是一种正则化手段, 将深度学习模型中的多分支结构的子路径随机"删除"。
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        # 获得随机删除路径后的结果
        out = drop_path(x, self.drop_prob, self.training)

        return out


# 归一化层 Layer Normalization
class LayerNorm(nn.Module):
    """
    LayerNorm 支持两种数据格式: channels_last (default) or channels_first.
    channels_last  对应的输入尺寸排序为：(batch_size, height, width, channels)
    channels_first 对应的输入尺寸排序为：(batch_size, channels, height, width)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()

        # 参数初始化
        self.eps = eps
        self.data_format = data_format
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)

        # 判断输入尺寸类型是否符合要求
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")

        # 获取输入形状
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # channels_last 输入尺寸
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        # channels_first 输入尺寸
        elif self.data_format == "channels_first":

            # 通道维度上求取均值和方差
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)

            # 计算归一化结果
            x = (x - mean) / torch.sqrt(var + self.eps)

            # 获取输出结果
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x


# 绘制 ROC 曲线
def roc_model(model, data_test, epochs):
    # 建立文件夹保存图片
    if not os.path.exists("./save_images"):
        os.mkdir("./save_images")

    # 预测标签和真实标签存储
    score_list = []
    label_list = []
    softmax = nn.Softmax(dim=1)  # 建立归一化模块

    for i, (test_images, test_labels) in enumerate(data_test):
        # 数据拆分
        test_img_1, test_img_2, test_img_3 = test_images[0], test_images[1], test_images[2]

        # 数据添加到CUDA
        test_img_1, test_img_2 = test_img_1.cuda(), test_img_2.cuda()
        test_img_3, test_labels = test_img_3.cuda(), test_labels.cuda()

        # 获取输出
        model_predict, _ = model(test_img_1, test_img_2, test_img_3, test_labels, epochs)

        # 获取综合预测值
        test_predicts = softmax(model_predict)

        # 存储预测值和真实值
        label_list.extend(test_labels.cpu().numpy())
        score_list.extend(test_predicts.detach().cpu().numpy())

    # 格式转换
    score_array = np.array(score_list)
    label_array = np.array(label_list)

    # ROC曲线
    fpr_dict, tpr_dict, _ = metrics.roc_curve(label_array, score_array[:, 1])
    roc_dict = metrics.auc(fpr_dict, tpr_dict)
    return fpr_dict, tpr_dict, roc_dict


# 绘制混淆矩阵
def confusion(model, data_test, epochs):
    # 建立文件夹保存图片
    if not os.path.exists("./save_images"):
        os.mkdir("./save_images")

    # 预测标签和真实标签存储
    score_list = []
    label_list = []

    for i, (test_images, test_labels) in enumerate(data_test):
        # 数据拆分
        test_img_1, test_img_2, test_img_3 = test_images[0], test_images[1], test_images[2]

        # 数据添加到CUDA
        test_img_1, test_img_2 = test_img_1.cuda(), test_img_2.cuda()
        test_img_3, test_labels = test_img_3.cuda(), test_labels.cuda()

        # 获取输出
        test_predicts, _ = model(test_img_1, test_img_2, test_img_3, test_labels, epochs)

        # 存储预测值和真实值
        label_list.extend(test_labels.cpu().numpy())
        score_list.extend(test_predicts.detach().cpu().max(1)[1])

    # 计算混淆矩阵
    cf_matrix = metrics.confusion_matrix(label_list, score_list)
    return cf_matrix


# 计算相关指标
def metrics_model(model, data_test, epochs):
    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签

    for i, (test_images, test_labels) in enumerate(data_test):
        # 数据拆分
        test_img_1, test_img_2, test_img_3 = test_images[0], test_images[1], test_images[2]

        # 添加到 CUDA
        test_img_1, test_img_2 = test_img_1.cuda(), test_img_2.cuda()
        test_img_3, test_labels = test_img_3.cuda(), test_labels.cuda()

        # 获取输出
        test_predicts, _ = model(test_img_1, test_img_2, test_img_3, test_labels, epochs)

        # 存储预测值和真实值
        label_list.extend(test_labels.cpu().numpy())
        score_list.extend(test_predicts.detach().cpu().max(1)[1])

    # 计算准确率
    accuracy_micro = metrics.accuracy_score(label_list, score_list)
    print('Accuracy is :', np.around(accuracy_micro, 4))

    # 计算召回率
    recall_micro = metrics.recall_score(label_list, score_list)
    print('Recall is :', np.around(recall_micro, 4))

    # 计算 F1 值
    f1score_micro = metrics.f1_score(label_list, score_list)
    print('F1score is :', np.around(f1score_micro, 4))

    # 计算精准率
    precision_micro = metrics.precision_score(label_list, score_list)
    print('Precision is :', np.around(precision_micro, 4))


# 计算 Mix-up
def mix_up_data(x, y, alpha=1.0, use_cuda=True):
    # 返回混合输入、目标对和 lambda
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# 计算 Mix-up 损失
def mix_up_criterion(y_a, y_b, lam):
    return lambda criterion, predict: lam * criterion(predict, y_a) + (1 - lam) * criterion(predict, y_b)


# 权重初始化
def init_weight(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_normal_(model.weight)
    if type(model) in [nn.Conv2d]:
        nn.init.kaiming_normal_(model.weight)
    if type(model) in [nn.BatchNorm2d]:
        nn.init.ones_(model.weight)


# 生成列表用于保存结果
def create_id(model, loader_test):
    # 生成列表用于保存结果
    save_prob = []
    save_path = []
    save_predict_label = []
    save_real_label = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (test_images, test_labels, file_name) in enumerate(loader_test):
            # 添加到CUDA中
            test_labels = torch.as_tensor(test_labels, dtype=torch.float32)
            test_images, test_labels = test_images.cuda(), test_labels.cuda()

            # 获取预测标签
            test_predicts = model(test_images)

            # 获取预测概率
            prob = softmax(test_predicts)

            # 获取预测标签
            predict_labels = test_predicts.detach().cpu().max(1)[1]

            # 保存上述结果
            save_path.extend(file_name)
            save_prob.extend(prob)

            save_real_label.extend(test_labels)
            save_predict_label.extend(predict_labels)

    return save_path, save_real_label, save_predict_label, save_prob


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        if alpha is None:
            alpha = [0.7, 0.15, 0.15]
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction, target):
        alpha = self.alpha[target].cuda()
        log_softmax = torch.log_softmax(prediction, dim=1)
        log_pt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        log_pt = log_pt.view(-1)
        ce_loss = -log_pt
        pt = torch.exp(log_pt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
