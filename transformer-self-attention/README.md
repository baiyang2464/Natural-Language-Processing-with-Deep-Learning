[TOC]

## 0.Attention Is All You Need

[attention is all you need](./attention-is-all-you-need.pdf)

主要的序列转换模型都是基于复杂的 RNN 或 CNN的encoder-decoder模型。表现最佳的模型也需通过注意力机制（attention mechanism）连接编码器和解码器。论文提出了一种新型的简单网络架构——Transformer，它完全基于注意力机制，准确的说是自注意self-attention机制，彻底放弃了循环和卷积

self-attention是一种会对一个单独的序列对不同位置进行关联的注意力机制。

## 1.transformer架构

沿用了encoder-decoder架构，框架用于解决由一个任意长度的源序列到另一个任意长度的目标序列的变换问题。即编码阶段将整个源序列编码成一个向量，解码阶段通过最大化预测序列概率，从中解码出整个目标序列。Transformer同样使用了encoder-decoder，在编码和解码阶段均使用stacked self-attention、和全连接。

### 1.1 encoder and decoder stacks



<p align="center">
	<img src=./pictures/transformer-model-architecture.png alt="Sample"  width="350">
	<p align="center">
		<em>transformer-model-architecture</em>
	</p>
</p>

**encoder堆叠了N=6层，每个encoder有两个子层**

+ multi-head self-attention layer
+ 全连接层

这两个子层内均是**residual connection**，之后再加上**layer normalization**，即，`LayerNorm(x+Sublayer(x))`（即图中的Add&Norm），Sublayer是子层自己实现的函数。为了方便这些residual connection，架构中的所有子层（包括embedding）,输出的维度均是d_model=512。这里的`d_model`指的是embedding的dim。所以输入比如有n个词，那就是一个`n×d_model`的矩阵。	

**decoder同样堆叠了N=6层，每个decoder有三个子层**

+ masked multi-head self-attention layer
+ encoder-decoder multi-head self-attention layer
+ 全连接层

每一子层均有residual connection包裹，之后再过一个layer normalization（图中Add & Norm）。

masked multi-head attention层，通过添加mask，这个子层不要encoder的输出作为输入，只要decoder 的output的embedding+pos_embedding作为输入。另外，还对self-attention层中的softmax进行了修改，由于输入的output embedding+pos_embedding有一个position的offset，所以结合masking，可以保证对位置i的预测，只依赖于位置小于i的位置的输出，让decoder只看到左边的词。

encoder-decoder multi-head self-attention layer用encoder最后的输出，一个`[sequence_len , d_model]`的矩阵，计算K，V，用上一层decoder的输出作为Q作为本层decoder的输入。

### 1.2 attention

attention函数：将一个query和一系列的(key, value)对映射起来。其中，query、key、value均是向量，维度分别为d_k, d_k, d_v。最终的输出是各value的加权组合：

<p align="center">
	<img src=./pictures/091401.svg alt="Sample"  width="550">
	<p align="center">
		<em> </em>
	</p>
</p>

+ scaled dot-product attention

query和所有key算点积，然后除以`√dk`，然后算softmax，得到的就是value的weight。实际应用中，一系列的query，记为矩阵`Q`，同样地，有`K`和`V`。

<p align="center">
	<img src=./pictures/091402.svg alt="Sample"  width="300">
	<p align="center">
		<em> </em>
	</p>
</p>

当d_model较大时，点积会变得特别大，所以softmax会出现有些区域的梯度变得特别小，因此，需要通过除以`√dk`来控制进行适度缩放。

### 1.3 multi-head attention

实际应用中，并不是直接计算`dmodel`维【前面提到了，架构中的所有子层（包括embedding）,输出的维度均是`dmodel=512`】的query，key，value。而是把这`dmodel`维拆成`h`份，所以`d_k=d_v=d_model/h`,(因为`d_model`是一样的，`h`也是一样的【论文中设为8】，所以`dk=dv=d_model/h=512/8=64`)。而这个变换通过对`Q,K,W`各自进行一个线性变换，变成`dk,dk,dv`维即可，最终通过concat(把`h=8`个结果首尾相连)，然后再做一个线性变换，变成`dmodel`维。

<p align="center">
	<img src=./pictures/091404.svg alt="Sample"  width="450">
	<p align="center">
		<em> </em>
	</p>
</p>

<p align="center">
	<img src=./pictures/091405.svg alt="Sample"  width="150">
	<p align="center">
		<em> </em>
	</p>
</p>

另外，attention在本模型中的应用

+ encoder-decoder结构中，对当前层的decoder而言，query来自前一层的decoder，key和value来自encoder的输出。
+ encoder有self-attention层。在self-attention中，所有层的key，value和query都来自于前一层encoder的输出。因此，当前层的encoder的每个位置可以(attend to)学习前一层的所有位置上的依赖关系。
+ 类似的，当前层的encoder的每个位置（例如位置i）可以attend to前一层的所有位置（包括位置i）。但为了保持auto-regressive特性，需要阻止leftword infomation（左边，即encoder的输出） 流入decoder，所以在scaled dot-product attention里使用mask把所有输入给softmax的 当前位置 i 之后的值都mask掉。

### 1.4  feed-forward networks

其实每个encoder和decoder层，除了attention子层之外，还有一个全连接的前馈网络。这个前馈网络会作用于每一个position。这个前馈网络包括了两种线性变换：

<p align="center">
	<img src=./pictures/091406.svg alt="Sample"  width="650">
	<p align="center">
		<em> </em>
	</p>
</p>
### 1.5 positional encoding

为了学习位置信息，为输入增加位置嵌入向量

在encoder和decoder的底部，也就是对应的embedding之后，加入了`d_model`维的position encoding，与embedding的维数一致，google是直接将词的embedding与position embedding 二者对应元素求和，facebook则是concat成一个长向量。

<p align="center">
	<img src=./pictures/091407.svg alt="Sample"  width="350">
	<p align="center">
		<em> </em>
	</p>
</p>
pos是词在序列中的绝对位置，i是位置向量的某个维度。因为对于任意固定的offset k，`PE（pos+k）`能被表示为`PE（pos）`的线性函数。即把id为`pos`的位置，映射成一个`d_model`维的向量，这个向量的第i个元素的数值是`PE(pos,i)`。

将每个位置编号，然后**每个编号对应一个向量**，通过结合位置向量和词向量，就给每个词都引入了一定的位置信息，这样Attention就可以分辨出不同位置的词了。

### 1.6 embeddings and softmax

与其他序列转换模型类似，本文也使用learned embeddings对input tokens和output tokens转换为`dmodel`维的向量（架构图中的 input embedding和output embedding）。

同时也使用learned linear变换和softmax将decoder的输出转换为预测的下一个token的概率。



## 2.为什么要self-attention

主要就序列到序列任务对比CNN和RNN

1. 能被并行化的计算量（可以用”**所需要的最小的序列操作**“来衡量）

能否并行化会直接较大程度的影响模型的训练时间

+ CNN是可以在一个卷积层内并行的对不同局部区域进行卷积的，所以Sequential Operations=O(1)
+ RNN此刻一个细胞的输入依赖于上一时刻细胞的输出，因此同一个序列内不能并行，只能串行，所以所以Sequential Operations=O(n)，n为一个序列种词或字的数目
+ self-attention在这一点上神似CNN，对序列中的每个词嵌入向量进行注意力的计算，同层内都是可以并行的，所以Sequential Operations=O(1)

<p align="center">
	<img src=./pictures/091402.png alt="Sample"  width="600">
	<p align="center">
		<em>并行的attention</em>
	</p>
</p>

2. 网络中长距离依赖的路径长度

这个主要是指如何学习序列中任意两个词嵌入向量之间的关系

+ RNN必须遍历一遍输入序列才能学习到序列中某两个嵌入向量的关系，所以Maximum Path Length=O(n)
+ CNN可以想象成N-gram模型，学习的是局部特征，若是想要学习到序列最远的嵌入向量之间的长距离依赖关系，可以通过分层卷积的方式，使得任意任意两个位置之间的长度距离是对数级别的O(logk(n))。

<p align="center">
	<img src=./pictures/091401.png alt="Sample"  width="500">
	<p align="center">
		<em>分层的卷积</em>
	</p>
</p>

+ self-attention是用每个嵌入向量的querry与序列内其它所有嵌入向量的<key,value>进行attention计算，使得学习到任意两个嵌入向量的长距离依赖关系复杂度为Maximum Path Length=O(1)

3. 每一层的总的计算复杂度

论文中这里没有太搞懂，暂不表述

4. self-attention还有一个好处，产出的模型更加可解释（interpretable）。

不仅是不同head的attention学到的task不同，而且从下图的visualization来看，甚至可以理解为，有的head学习某种句法/语法关系，有的学习另一种句法/语法关系。

<p align="center">
	<img src=./pictures/091403.png alt="Sample"  width="600">
	<p align="center">
		<em>分层的卷积</em>
	</p>
</p>

总的对比

<p align="center">
	<img src=./pictures/091404.png alt="Sample"  width="600">
	<p align="center">
		<em>CNN\RNN\self-attention对比</em>
	</p>
</p>

**总结**：self-attention既可以并行学习，又可以较方便的学习到长距离的依赖关系，加入“多头自注意”机制还可以学习到不同种类的依赖关系。

## 3.transformer总体过程概述

