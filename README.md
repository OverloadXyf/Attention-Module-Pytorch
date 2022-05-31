# Attention-Module-Pytorch

## Attention-Module 包括

* SEnet: Squeeze-and-Excitation Networks.   Paper Address ---> https://arxiv.org/pdf/1709.01507.
* ECAnet: Efficient Channel Attention for Deep Convolutional Neural Networks. Paper Address ---> https://arxiv.org/pdf/1910.03151.pdf
  * 自适应卷积核大小（Adaptive Kernel Size）
  * 固定卷积核大小（Kernerl Size=3）
* CBAM: Convolutional Block Attention Module. Paper Address ---> https://arxiv.org/pdf/1807.06521.pdf
  * 使用Channel Attention (CA)或者Spatial Attention (SA),或者 CA&SA
  * 使用Global Average Pooling (GAP), Global Maximum Pooling (GMP),或GAP&GMP
  * Channel Pooling使用Mean, Max，或者Mean&Max

### 即插即用 非常方便

* input=tensor(1,256,28,28)
| Attention Module |   Flops   | Params |
| :----------------: | :---  ----: | :---_---: |
|      SEnet      | 209.18K |    8.46K |
|      ECAnet k=3| 207.47K |    3  |
|      ECAnet (Adaptive_k) | 201.98 | 5 |
|      CBAM_Channel_2pool_Spatial_2pool  | 497.55k | 8.56K |
