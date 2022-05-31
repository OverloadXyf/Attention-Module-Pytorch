import torch
import torch.nn as nn
import torch.nn.functional as F

'''
"CBAM: Convolutional Block Attention Module":https://arxiv.org/pdf/1807.06521.pdf

CHI:卷积模块的注意力机制 属于混合注意力机制网络（空间和通道）
CHI:在这里 模型可以有多种构建方式以探索不同结构之间的差异
CHI:可以设置是否使用Channel Attention和Spatial Attention 此处3种方式
CHI:可以设置Channel Attention是使用GAP,GMP,或GAP&GMP 此处3种方式
CHI:可以设置Channel Pool是使用Mean,Max,或Mean&Max 此处3种方式
CHI:共计3×3×3=27种方式

ENG: the attention mechanism of convolution module belongs to hybrid attention mechanism network (space and channel)  
ENG: here, the model can be built in a variety of ways to explore the differences between different structures  
ENG: you can set whether to use channel attention and spatial attention  
ENG: channel attention can be set to use gap, gmp, or gap&gmp  
ENG: you can set the channel pool to use mean, max, or mean&max  
ENG: 27 modes in total
'''
class ChannelGate(nn.Module):
    def __init__(self,channel,reduction=16,chan_Att_type=['Avg','Max']):
        super(ChannelGate, self).__init__()

        self.c_type=chan_Att_type
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.mlp=nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel),
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        b,c,_,_=x.size()

        att_sum=None
        for type in self.c_type:

            if 'Avg' == type:
                att_avg=self.avgpool(x).view(b,c)
                att_raw=self.mlp(att_avg).view(b,c,1,1)

            elif 'Max' == type:
                att_max=self.maxpool(x).view(b,c)
                att_raw=self.mlp(att_max).view(b,c,1,1)

            else:
                raise ValueError('Channel Pooling Type Error. Got{}'.format(type))

            if att_sum is None:
                att_sum=att_raw
            else:
                att_sum=att_sum+att_raw

        scale=self.sigmoid(att_sum).expand_as(x)

        return x*scale

class ChannelPool(nn.Module):
    def __init__(self,spatial_Suppress=['Mean','Max']):
        super(ChannelPool, self).__init__()
        self.suppress_type=spatial_Suppress

    def forward(self,x):
        if ('Mean' in self.suppress_type) and ('Max' in self.suppress_type):
            c_mean=torch.mean(x,dim=1).unsqueeze(1)
            c_max=torch.max(x,dim=1)[0].unsqueeze(1)
            return torch.cat([c_mean,c_max],dim=1)
        elif 'Mean' in self.suppress_type:
            return torch.mean(x,dim=1).unsqueeze(1)
        elif 'Max' in self.suppress_type:
            return torch.max(x,dim=1)[0].unsqueeze(1)

class SpatialGate(nn.Module):
    def __init__(self,spatial_Suppress=['Mean','Max']):
        super(SpatialGate, self).__init__()

        self.suppress=spatial_Suppress
        self.compress=ChannelPool(spatial_Suppress)

        mul=len(spatial_Suppress)
        k_size=7
        self.spatial=nn.Sequential(
            nn.Conv2d(mul,1,kernel_size=k_size,stride=1,padding=k_size//2),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01,affine=True),
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        compress=self.compress(x)
        temp=self.spatial(compress)
        scale=self.sigmoid(temp)

        return x*scale



class CBAM(nn.Module):
    def __init__(self,channel,reduction=16,chan_Att_type=['Avg','Max'],spatial_Suppress=['Mean','Max'],use_channel=True,use_spatial=True):
        '''

        :param chan_Att_type:
        :param spatial_Suppress:
        '''
        super(CBAM, self).__init__()

        self.chan_type=chan_Att_type
        self.spatial_sup_type=spatial_Suppress
        self.channel=channel
        self.reduction=reduction
        self.use_channel=use_channel
        self.use_spatial=use_spatial
        self.ChannelAtt=ChannelGate(channel,chan_Att_type=self.chan_type) if use_channel else nn.Sequential()
        self.ChannelPool=ChannelPool(self.spatial_sup_type) if use_spatial else nn.Sequential()
        self.SpatialAtt=SpatialGate(self.spatial_sup_type) if use_spatial else nn.Sequential()

    def forward(self,x):

        if self.use_channel:
            chan_feat=self.ChannelAtt(x) # N C H W
        else:
            chan_feat=x # N C H W

        if self.use_spatial:
            spatial_feat=self.SpatialAtt(chan_feat) # N 1 H W
        else:
            spatial_feat=chan_feat


        return chan_feat*spatial_feat

if __name__ == '__main__':

    # model=CBAM(32,chan_Att_type=['Avg'],spatial_Suppress=['Mean'],use_channel=False)
    model=CBAM(32)
    a=torch.rand((1,32,64,64))
    res=model(a)
    print(res.shape)


