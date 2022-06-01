import torch
import torch.nn as nn
import torch.nn.functional as F

'''
"GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond":https://arxiv.org/abs/1904.11492?context=cs.LG
"Non-local Neural Networks":https://arxiv.org/abs/1711.07971v1
CHI:GCnet是对Non-local networks的简化，这里的SE网络的实现不同于原始SE代码。我们将在另一个仓库中作为SE变体来讨论。
ENG:GCnet is a simplification of non local networks. The implementation of SEnetwork here is different from the 
    original SE code.We will discuss it as an SE variant in another repository.
'''

class GCLayer(nn.Module):

    def __init__(self,channel,hidden_channel=None,pool='spatial_pool',fusion=['add']):
        '''
        Global Context Networks: pool & fusion model can be modified
        :param channel: input&output channels
        :param hidden_channel: hidden channels, if None, hidden channels=channels
        :param pool: Optional-->'spatial_pool' or 'avg' pool
        :param fusion: channel 'add' or 'mul'
        '''
        super(GCLayer, self).__init__()

        assert pool in ['spatial_pool','avg']
        assert all([f in ['add','mul'] for f in fusion])
        assert len(fusion)>0,'At least one fusion mode. We Recommend mul'
        self.channel=channel
        if hidden_channel is None:
            hidden_channel=channel
        self.hidden_channel=hidden_channel
        self.pool=pool
        self.fusion=fusion

        if 'spatial pool' in pool:
            self.conv_mask=nn.Conv2d(channel,1,kernel_size=1)
            self.softmax=nn.Softmax(dim=2) # Softmax for H*W
        else:
            self.avg_pool=nn.AdaptiveAvgPool2d(1)

        if 'add' in fusion:
            self.channel_add_conv=nn.Sequential(
                nn.Conv2d(self.channel,self.hidden_channel,kernel_size=1),
                nn.LayerNorm([self.hidden_channel,1,1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_channel,self.channel,kernel_size=1)
            )
        else:
            self.channel_add_conv=None

        if 'mul' in fusion:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.channel, self.hidden_channel, kernel_size=1),
                nn.LayerNorm([self.hidden_channel, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_channel, self.channel, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self,x):
        b,c,h,w=x.size()
        if self.pool=='spatial pool':
            input_x=x
            input_x=input_x.view(b,c,h*w)
            input_x=input_x.unsqueeze(1)             # N 1 C H*W Original Info Flatten

            context_mask=self.conv_mask(x)           # N 1 H W
            context_mask=context_mask.view(b,1,h*w)  # N 1 H*W
            context_mask=self.softmax(context_mask)  # N 1 H*W Softmax Confidence/Importance
            context_mask=context_mask.unsqueeze(3)   # N 1 H*W 1

            context=torch.matmul(input_x,context_mask) # N 1 C 1
            context=context.view(b,c,1,1)            # N C 1 1
        else:
            context=self.avg_pool(x)                  # N C 1 1

        return context


    def forward(self,x):
        context=self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            channel_mul_term=torch.sigmoid(self.channel_mul_conv(context))
            out=x*channel_mul_term

        else:
            out=x


        if self.channel_add_conv is not None:
            channel_add_term=self.channel_add_conv(context)
            out=out+channel_add_term


        return out

if __name__ == '__main__':
    model=GCLayer(3)
    a=torch.rand(1,3,25,25)
    res=model(a)
    print(res.shape)

