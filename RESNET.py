import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from numpy import linalg as LA


class ResNet:
    def __init__(self):
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_feat(self, img_path):
        # 加载图像并进行预处理
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        # 添加一个批次维度
        img = img.unsqueeze(0)
        # 使用ResNet模型提取特征
        with torch.no_grad():
            features = self.model(img)
        # 将特征张量展平并进行归一化
        norm_feature = features[0] / LA.norm(features[0])
        return np.array(norm_feature.reshape(-1))
