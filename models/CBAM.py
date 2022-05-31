import torch
import torch.nn as nn
import torch.nn.functional as F

'''

'''
class ChannelGate(nn.Module):
    def __init__(self,channel,reduction=16,chan_Att_type=['Avg','Max']):
        super(ChannelGate, self).__init__()

        self.c_type=chan_Att_type
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.maxpool=nn.AdaptiveMaxPool2d(2)
        self.mlp=nn.Sequential(
            nn.Linear(channel,channel//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,_,_=x.size()

        att_sum=None
        for type in self.c_type:

            if 'Avg' ==type:
                att_avg=self.avgpool(x).view(b,c)
                att_raw=self.mlp(att_avg).view(b,c,1,1)



class CBAM(nn.Module):
    def __init__(self,channel,reduction=16,chan_Att_type=['Avg','Max'],spatial_Suppress=['Mean','Max']):
        '''

        :param chan_Att_type:
        :param spatial_Suppress:
        '''
        super(CBAM, self).__init__()

        self.chan_type=chan_Att_type
        self.spatial_sup_type=spatial_Suppress
        self.channel=channel
        self.reduction=reduction



