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
* GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond. Paper Address ---> :https://arxiv.org/abs/1904.11492?context=cs.LG
  * 融合（fusion）方式包括直接add和mul

### 即插即用 非常方便

  * input=tensor(1,256,28,28)

| Attention Module |   Flops   | Params |
| :----------------: | :----------: | :--------: |
|      SEnet      | 209.18K |    8.46K |
|      ECAnet (k=3)| 207.47K |    3  |
|      ECAnet (Adaptive_k) | 201.98K | 5 |
|      CBAM_Channel_2pool_Spatial_2pool  | 497.55K | 8.56K |
|      GCnet fusion=add | 332.54K| 132.1K |
|      GCnet fusion=mul | 332.54K| 132.1K |
|      GCnet fusion=mul&add | 464.38K| 264.19K |
