import torch
from ptflops import  get_model_complexity_info
from models.SEnet import SELayer
from models.CBAM import CBAM
from models.ECAnet import ECALayer,ECALayer_Adaptive_Kernel

with torch.cuda.device(0):


    # net=SELayer(256)
    # net=ECALayer()
    # net=CBAM(256)
    net=ECALayer_Adaptive_Kernel(256)

    macs,params=get_model_complexity_info(net,(256,28,28),as_strings=True,print_per_layer_stat=True,verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity:',macs))
    print('{:<30}  {:<8}'.format('Number of parameters:',params))