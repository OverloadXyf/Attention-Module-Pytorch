import torch
import torch.nn as nn
import torch.nn.functional as F

'''
"Squeeze-and-Excitation Networks":https://arxiv.org/pdf/1709.01507.pdf
CHI:标准的搭建模式，其中的参数可以修改
ENG:Standard setup mode, in which parameters can be modified
'''

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer, self).__init__()
#
#         self.avg_pool = nn.AdaptiveMaxPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         # 线性层 最后会是一个sigmoid 输出的相当于attention
#         '''2022/5/6 多了一個池化所以輸入維度增加'''
#         self.fc = nn.Sequential(
#             nn.Linear(2 * channel, channel // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y1 = self.avg_pool(x).view(b, c)
#         y2 = self.max_pool(x).view(b, c)
#         '''2022/5/6 多了max_pool 在通道上堆疊起來'''
#         y = torch.cat([y1, y2], dim=-1)
#         # y 相当于输出的是一个注意力attention 乘以原来的tensor 那就是对Channel的注意力变换
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y