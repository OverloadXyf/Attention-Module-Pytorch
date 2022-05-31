# Attention-Module-Pytorch

## Attention-Module 包括

* SEnet: Squeeze-and-Excitation Networks.   Paepr Address ---> https://arxiv.org/pdf/1709.01507.
* ECAnet: Efficient Channel Attention for Deep Convolutional Neural Networks. Paepr Address ---> https://arxiv.org/pdf/1910.03151.pdf
    * 自适应卷积核大小（Adaptive Kernel Size）
    * 固定卷积核大小（Kernerl Size=3）
* CBAM: Convolutional Block Attention Module. Paepr Address ---> https://arxiv.org/pdf/1807.06521.pdf
    * 使用Channel Attention (CA)或者Spatial Attention (SA),或者 CA&SA
    * 使用Global Average Pooling (GAP), Global Maximum Pooling (GMP),或GAP&GMP
    * Channel Pooling使用Mean, Max，或者Mean&Max 

### 即插即用 非常方便
