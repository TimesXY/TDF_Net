# 导入库函数
import cv2 as cv
import numpy as np

# 获取目标层的梯度
import torch


class ActivationsAndGradients:
    def __init__(self, model, target_layers, reshape_transform):

        # 参数初始化
        self.handles = []
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform

        # 保存前向传播和反向传播的特征图和激活值
        for target_layer in target_layers:

            # 获取前向传播的特征图或激活值
            self.handles.append(target_layer.register_forward_hook(self.save_activation))

            # 获取反向传播的特征图或激活值，实现版本兼容功能
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))
            else:
                self.handles.append(target_layer.register_backward_hook(self.save_gradient))

    # 实现前向过程特征图的存储
    def save_activation(self, module, input, output):

        # 获取输出特征图
        activation = output

        # 判断是否为 Transformer 模型，若是则进行特征图重组
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)

        # 非 Transformer模型则直接添加到列表
        self.activations.append(activation.cpu().detach())

    # 实现梯度的保存，梯度是按相反顺序计算
    def save_gradient(self, module, grad_input, grad_output):

        # 获取输出梯度
        grad = grad_output[0]

        # 判断是否为 Transformer 模型，若是则进行特征图重组
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)

        # 非 Transformer模型则直接添加到列表，添加在前面
        self.gradients = [grad.cpu().detach()] + self.gradients

    # 前向计算过程，返回模型输出
    def __call__(self, x1, x2, x3):
        self.gradients = []
        self.activations = []
        out_model, _ = self.model(x1, x2, x3, torch.tensor(1).cuda(), 150)
        return out_model, _

    # 释放函数，使用完后需要释放 hook
    def release(self):
        for handle in self.handles:
            handle.remove()


# grad_cam_sk 类激活图
class GradCAM:
    def __init__(self, model, target_layers, reshape_transform=None, use_cuda=False):

        """
        大多数的突出性归因论文中，突出性是通过单一的目标层计算的。
        通常，它是最后一个卷积层。在这里，我们支持传递一个包含多个目标层的列表。
        它将为每张图像计算出显著性图像，然后对它们进行汇总（使用默认的平均汇总）。
        具有更多的灵活性，例如使用所有的卷积层，例如所有的BN层或其他。
        """

        # 参数初始化
        self.cuda = use_cuda
        self.model = model.eval()
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform

        # 判断是否调用GPU
        if self.cuda:
            self.model = model.cuda()

        # 获取保存的正向激活和反向梯度
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform)

    # 获取特征图对应CAM通道权重
    @staticmethod
    def get_cam_weights(grads):
        return np.mean(grads, axis=(2, 3), keepdims=True)

    # 获取损失函数值，预测类被和真实类别的误差累计
    @staticmethod
    def get_loss(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    # 获取 CAM 图像，通道上取均值求和
    def get_cam_image(self, activations, grads):
        weights = self.get_cam_weights(grads)
        weighted_activations = weights * activations
        cam = weighted_activations.sum(axis=1)
        return cam

    # 获取目标层的特征图高宽
    @staticmethod
    def get_target_width_height(input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    # 计算网络每层的 CAM 图像
    def compute_cam_per_layer(self, input_tensor):

        # 循环获取每层网络的前向输出，组成列表
        activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]

        # 循环获取每层网络的反向梯度，组成列表
        grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]

        # 获取目标层的特征图高宽
        target_size = self.get_target_width_height(input_tensor)

        # 循环获取网络每层的前向输出和反向梯度
        cam_per_target_layer = []
        for layer_activations, layer_grads in zip(activations_list, grads_list):
            # 获取 CAM 图像，通道上取均值求和
            cam = self.get_cam_image(layer_activations, layer_grads)

            # 对 CAM 图像进行 ReLu 激活
            cam[cam < 0] = 0

            # 对图像进行放缩和归一化
            scaled = self.scale_cam_image(cam, target_size)

            # 添加图像到列表结果
            cam_per_target_layer.append(scaled[:, None, :])

        # 返回的 CAM 图像
        return cam_per_target_layer

    # 聚合多层的结果？（各通道，各层网络）
    def aggregate_multi_layers(self, cam_per_target_layer):

        # 对特征图进行拼接
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)

        # 返回特征图中的最大值
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)

        # 多个目标层特征图的平均聚集
        result = np.mean(cam_per_target_layer, axis=1)

        # 对图像进行放缩和归一化
        return self.scale_cam_image(result)

    # 特征图的放缩和归一化
    @staticmethod
    def scale_cam_image(cam, target_size=None):

        # 目标结果
        result = []
        for img in cam:

            # CAM 结果的归一化
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))

            # 将 CAM 放缩到目标大小
            if target_size is not None:
                img = cv.resize(img, target_size)

            # 多张图像的合并
            result.append(img)

        # 返回结果值
        result = np.float32(result)
        return result

    # 前向计算函数
    def __call__(self, input_tensor1, input_tensor2, input_tensor3, target_category=None):

        # 判断是否采用 GPU
        if self.cuda:
            input_tensor1 = input_tensor1.cuda()
            input_tensor2 = input_tensor2.cuda()
            input_tensor3 = input_tensor3.cuda()

        # 正向传播得到网络输出 logit (未经过softmax)
        output, _ = self.activations_and_grads(input_tensor1, input_tensor2, input_tensor3)

        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor1.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
            print(f"category id: {target_category}")
        else:
            assert (len(target_category) == input_tensor1.size(0))

        # 梯度清零
        self.model.zero_grad()

        # 计算梯度
        loss = self.get_loss(output, target_category)

        # 梯度反传
        loss.backward(retain_graph=True)

        # 获取不同层的 CAM
        cam_per_layer = self.compute_cam_per_layer(input_tensor1)

        # 均值求取最终的 CAM
        return self.aggregate_multi_layers(cam_per_layer)

    # 释放函数
    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    # 索引报错模块
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True


# CAM 图像和原始图像的显示
def show_cam_on_image(img: np.ndarray, mask: np.ndarray, use_rgb: bool = False,
                      colormap: int = cv.COLORMAP_JET) -> np.ndarray:
    """
     这个函数将 CAM 以热图的形式叠加在图像上。
     默认情况下，热图是 BGR 格式的。
     param img : RGB 或 BGR 格式的基本图像。
     param mask: CAM
     param use_rgb : 是否使用 RGB 或 BGR 热图，如果图像是RGB格式，应该设置为True。
     param colormap: 要使用的 OpenCV 颜色映射。
     return: 带有CAM覆盖的默认图像。
    """

    # 获取热图 JET
    heatmap = cv.applyColorMap(np.uint8(255 * mask), colormap)

    # 判断是否采用RGB图像
    if use_rgb:
        heatmap = cv.cvtColor(heatmap, cv.COLOR_BGR2RGB)

    # 热力图的归一化
    heatmap = np.float32(heatmap) / 255

    # 结果放缩到 0-1 之间
    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    # 对原图和热力图进行叠加
    cam = heatmap + img

    # 再次放缩到 0 - 1
    cam = cam / np.max(cam)

    # 返回 uint8 图像
    return np.uint8(255 * cam)


# 图像的中心裁剪
def center_crop_img(img: np.ndarray, size: int):
    # 获取图像的长宽高
    h, w, _ = img.shape

    # 长宽一致，则直接返回
    if w == h == size:
        return img

    # 若长宽不一致，则进行裁剪
    if w < h:
        ratio = size / w
        new_w = size
        new_h = int(h * ratio)
    else:
        ratio = size / h
        new_h = size
        new_w = int(w * ratio)

    # 图像按照比例放缩
    img = cv.resize(img, dsize=(new_w, new_h))

    # 对放缩后的结果进行裁剪
    if new_w == size:
        h = (new_h - size) // 2
        img = img[h: h + size]
    else:
        w = (new_w - size) // 2
        img = img[:, w: w + size]

    return img
