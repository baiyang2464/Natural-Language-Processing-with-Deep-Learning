## 深度学习网络中的Normalization

+ 为什么要进行Normalization

一种是说normalization能够解决“Internal Covariate Shift（内部协方差漂移）”这种问题。简单理解就是随着层数的增加，中间层的输出的分布会发生“漂移”。另外一种说法是：normalization能够解决梯度弥散。通过将输出进行适当的缩放，可以缓解梯度消失的状况。

+ 在什么位置进行Normalization

一般说在仿射变换之后，激活函数之前

## CV中的Normalization

以CV为例介绍Normalization，一个Batch的图像数据shape为[样本数N, 通道数C, 高度H, 宽度W]，是四维的，将其最后两个维度flatten，得到的是[N, C, H*W]以方便表示

  <p align="center">
	<img src=./pictures/091101.png alt="Sample"  width="900">
	<p align="center">
		<em> </em>
	</p>

图片出至[何恺明Group Normalization](<https://arxiv.org/pdf/1803.08494.pdf>)

翻译一下图片下面那段话：归一化方法：每个子图表示一个特征映射的张量，每个张量中用N表示batch axis（数值代表batch size），C表示channel axis，（H,W）表示spatial axes（理解为一张图一个通道中的所有像素空间）。蓝色的像素表示使用相同的均值和方差进行归一化，这些均值和方差也是在标蓝的像素里计算的。

### 0.总览

输入的图像shape记为[N, C, H, W]，这几个方法主要的区别就是在：

batchNorm是在所有实例对应通道上，对N、H、W维度做归一化，对小batchsize效果不好；
layerNorm是只考虑单个实例在所有通道方向上，对C、H、W维度归一化，主要对RNN作用明显；
instanceNorm在图像像素上（单个实例的单个通道上），对H、W维度做归一化，用在风格化迁移；
GroupNorm单个实例对channel分组，然后对channel组内的像素做归一化。

### 1.Batch Normalization

#### 1.1 算法过程及代码

算法过程：

+ 沿着通道计算每个batch的均值u
+ 沿着通道计算每个batch的方差σ^2
+ 对x做归一化，x’=(x-u)/开根号(σ^2+ε)
+ 加入缩放和平移变量γ和β ,归一化后的值，y=γx’+β

  <p align="center">
	<img src=./pictures/091103.png alt="Sample"  width="350">
	<p align="center">
		<em> </em>
	</p>

加入缩放平移变量的原因是：直接标准归一化后使样本聚集在坐标原点附近，若后接激活函数的话，激活函数在原点附近是近似线性变换的，会消弱非线性的学校能力。 这两个参数是可学习得到参数。

用在4维数据输入情况[btach-size , channel-size , picture-height , picture-width]的代码：

```python
import numpy as np

def Batchnorm(x, gamma, beta, bn_param):

    # x_shape:[B, C, H, W]  
    running_mean = bn_param['running_mean']
    running_var = bn_param['running_var']
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(0, 2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta

    # 因为在测试时是单个图片测试，这里保留训练时的均值和方差，用在后面测试时用
    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return results, bn_param
```

注意：在训练过程中的batch norm层会对每次求得的均值和方法取滑动平均，以留作预测时候使用

#### 1.2 特点或不足

+ 对batchsize的大小比较敏感：如果batch_size太小，则bn效果下降明显
+ RNN 等动态网络使用 BN 效果不佳：究其原因，RNN的输入是序列数据，上一时间步和下一时间步的数据可能是某种顺序逻辑关系，不像CV中一批数据中图片对应通道存在分布相似关系

+ 训练时和推理时统计量不一致：预测时输入单个实例，不存在批量，所以用的均值和方差是来自训练时的滑动平均

### 2.Layer Normalization

BN与LN的区别在于：

+ LN中同层神经元输入拥有相同的均值和方差，不同的输入样本有不同的均值和方差；
  BN中则针对不同神经元输入计算均值和方差，同一个batch中的输入拥有相同的均值和方差。

+ 所以，LN不依赖于batch的大小和输入sequence的深度，因此可以用于batchsize为1和RNN中对边长的输入sequence的normalize操作。

+ LN用于RNN效果比较明显，但是在CNN上，不如BN。

  <p align="center">
	<img src=./pictures/091105.png alt="Sample"  width="350">
	<p align="center">
		<em> </em>
	</p>

用在4维数据输入情况[btach-size , channel-size , picture-height , picture-width]的代码：

```python
def Layernorm(x, gamma, beta):

    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(1, 2, 3), keepdims=True)
    x_var = np.var(x, axis=(1, 2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```

### 3.Instance Normalization

Instance Normalization是对每个实例的每个通道单独进行归一化。不依赖于batch

图像风格化中，生成结果主要依赖于某个图像实例，所以对整个batch归一化不适合图像风格化中，因而对HW做归一化。可以加速模型收敛，并且保持每个图像实例之间的独立。

```python
def Instancenorm(x, gamma, beta):

    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5

    x_mean = np.mean(x, axis=(2, 3), keepdims=True)
    x_var = np.var(x, axis=(2, 3), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```

### 4.Group Normalization

bn和in都走了两个极端，且不想依赖于batch size可以用group normalization

GN将channel方向分group，然后每个group内做归一化，算`(C//G)*H*W`的均值，这样与batchsize无关，不受其约束。

```python
def GroupNorm(x, gamma, beta, G=16):

    # x_shape:[B, C, H, W]
    results = 0.
    eps = 1e-5
    x = np.reshape(x, (x.shape[0], G, x.shape[1]/16, x.shape[2], x.shape[3]))

    x_mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(x, axis=(2, 3, 4), keepdims=True0)
    x_normalized = (x - x_mean) / np.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results
```

## data mining 中的Normalization

数据挖掘输入的是一个数据矩阵，一个batch的数据是B={x1,x2,..,xm}，batch size=m，每条数据xi有n维的特征，`xi=[xi^(1),xi^(2),...,xi^(n)]`

那么一个特征维度就对应CV图片里面的channel，features-nums等于channel-size，所以数据挖掘中的bn做法是在一批数据中对应维度内独立的进行归一化

+ 沿着每列特征计算每个batch的均值u
+ 沿着每列特征计算每个batch的方差σ^2
+ 对x做归一化，x’=(x-u)/开根号(σ^2+ε)
+ 加入缩放和平移变量γ和β ,归一化后的值，y=γx’+β

  <p align="center">
	<img src=./pictures/091102.png alt="Sample"  width="350">
	<p align="center">
		<em> </em>
	</p>

  <p align="center">
	<img src=./pictures/091104.png alt="Sample"  width="350">
	<p align="center">
		<em> </em>
	</p>

## NLP中的Normalization

NLP中一个batch有batch_size个sequence，每个sequence有sequence_len个word。

则一个sequence就是一个实例，对应cv中的一张图片

一个sequence中的每个word就相当于图片中的一个channel。

## 论文链接

1.Batch Normalization

https://arxiv.org/pdf/1502.03167.pdf

2.Layer Normalizaiton

https://arxiv.org/pdf/1607.06450v1.pdf

3.Instance Normalization

https://arxiv.org/pdf/1607.08022.pdf

https://github.com/DmitryUlyanov/texture_nets

4.Group Normalization

https://arxiv.org/pdf/1803.08494.pdf
