import torch
import torch.nn as nn
import math
import torch.nn.functional as F

'''
"ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks":https://arxiv.org/pdf/1910.03151.pdf
1.ECALayer_Adaptive_Kernel:自适应的卷积核
2.ECALayer：卷积核大小固定为3
'''

class ECALayer_Adaptive_Kernel(nn.Module):
    def __init__(self, channel,b=1,gamma=2):
        '''
        CHI:这里是文中提出的自适应卷积和的实现
        ENG:Here is the implementation of the adaptive convolution kernel proposed in this paper
        :param channel:input&output channels
        '''
        super(ECALayer_Adaptive_Kernel, self).__init__()

        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs(math.log(channel, 2) + b) / gamma)
        k_size = t if t % 2 else t + 1


        self.fc = nn.Sequential(
            nn.Conv1d(1,1,kernel_size=k_size,padding=k_size//2,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        temp = self.avg_pool(x) # N C 1 1
        temp= self.fc(temp.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1) # N,C,1,1
        return x * temp.expand_as(x)

class ECALayer(nn.Module):
    def __init__(self,k_size=3):
        '''
        CHI:固定卷积核大小为3
        ENG:kernel_size=3
        :param channel:input&output channels
        '''
        super(ECALayer, self).__init__()

        self.avg_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(1,1,kernel_size=k_size,padding=k_size//2,bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        temp = self.avg_pool(x) # N C 1 1
        temp= self.fc(temp.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1) # N,C,1,1
        return x * temp.expand_as(x)


