

### 一、word2vec

#### 1.cbow模型

  <p align="center">
	<img src=./pictures/090701.png alt="Sample"  width="550">
	<p align="center">
		<em> </em>
	</p>


用窗口内目标词的上下文词来预测目标词

**（1）算法是希望计算一个概率![](./pictures/090806.png)，并希望这个概率越大越好**

  <p align="center">
	<img src=./pictures/090802.png alt="Sample"  width="600">
	<p align="center">
		<em> </em>
	</p>
</p>

其中m是窗口尺寸，![](./pictures/090807.png)是最后预测出的一个向量，维度是整个词典的大小，而用上下文预测出其它词的概率$\hat{y_t}^{(i)}=P(w_i|w_{t-m},...,w_{t+m})\  i\neq t$越小越好

引入对数似然函数，然后取反作为损失函数

  <p align="center">
	<img src=./pictures/090810.png alt="Sample"  width="300">
	<p align="center">
		<em> </em>
	</p>

而又![](./pictures/090808.png)是word_target的onehot表示的，因此可以简化写为

  <p align="center">
	<img src=./pictures/090811.png alt="Sample"  width="400">
	<p align="center">
		<em> </em>
	</p>
</p>

SGD的目标函数来达到最大化![](./pictures/090806.png) ，最小化![](./pictures/090812.png)的目标

  <p align="center">
	<img src=./pictures/090813.png alt="Sample"  width="400">
	<p align="center">
		<em> </em>
	</p>
</p>

其中![](./pictures/090809.png)是cbow模型的输出向量，![](./pictures/090807.png)是输出向量经过softmax后的预测向量,![](./pictures/090808.png)是word_target的onehot表示

**（2）概率![](./pictures/090806.png) 的计算方法**

   <p align="center">
	<img src=./pictures/090814.png alt="Sample"  width="500">
	<p align="center">
		<em> </em>
	</p>
</p>

其中![](./pictures/090809.png)是cbow模型的输出向量

在tensorflow中tf.nn.softmax是把一个向量中的元素进行概率分布化，得到的仍是一个与softmax输入维度相同的概率分布向量

```shell
>>> A=[[1.0,2.0,3.0],[1.0,5.0,5.0],[9.0,8.0,7.0]]
>>> with tf.Session() as sess: 
...     print(sess.run(tf.nn.softmax(A)))
... 
[[0.09003057 0.24472848 0.66524094]
 [0.00907471 0.4954626  0.4954626 ]
 [0.66524094 0.24472848 0.09003057]]

```

而数学公式里面的softmax则是输出一个概率值，所以会导致表示不一样

在实际中可以将cbow模型中间层经仿射变换后过一个softmax，得到上下文预测出目标词的概率，用v_t标乘上这个概率向量，就取出了这个概率

  <p align="center">
	<img src=./pictures/090815.png alt="Sample"  width="350">
	<p align="center">
		<em> </em>
	</p>
</p>

举例：x_t为某个单词onehot表示的列向量[0,0,0...,0,1,0,...,0]^T

#### 2.skip-gram模型

  <p align="center">
	<img src=./pictures/090803.png alt="Sample"  width="600">
	<p align="center">
		<em> </em>
	</p>
</p>

用窗口内的目标词来预测上下文词，让预处上下文词的概率极大化，以此来得到目标词的词向量

可以写出目标函数

  <p align="center">
	<img src=./pictures/090801.png alt="Sample"  width="550">
	<p align="center">
		<em> </em>
	</p>
</p>
#### 3.负样本

对朴素的word2vec方法改进，对无关的词进行惩罚

（1）将需要计算全局的softmax改成二分类的sigmoid

![](./pictures/090816.png) 

就变成

![](./pictures/090817.png)

即将多分类改成二分类

（2）同时增加对非窗口词的负采样

目标词的上下文窗口内的词抽样称为正样本，窗口外的词的抽样则为负样本

现在目标变成：最大化目标词窗口内真实单词出现的概率，最小化随机单词出现在目标词周围的概率

目标函数就可以变成：

  <p align="center">
	<img src=./pictures/090818.png alt="Sample"  width="550">
	<p align="center">
		<em> </em>
	</p>
</p>

+ 从窗口外随机抽取k个词作为负样本
+ 对每个词出现的频率进行3/4的幂函数
  + 即每次词抽取的概率为![](./pictures/090819.png)，U(w)为词w的频数
  + 这种幂指数使频率较低的单词也可被更频繁地采样

### 二、glove词向量

+ 问题：词义相近的词对贡献次数多，词义差得比较远的词对共现次数比较少，但其实他们的区分度并不明显。能否用共现之间的比值来增大区分度？

+ Motivation：对word-pair cooccur进行建模，拟合不同word-pair的cooccur之间的差异。

用矩阵变换来模拟共现矩阵中的统计规律

共现矩阵：用于统计词对同时出现的概率的矩阵

统计规律：不同词对同时出现的概率比值

目标函数如下：

  <p align="center">
	<img src=./pictures/090804.png alt="Sample"  width="350">
	<p align="center">
		<em> </em>
	</p>
</p>

其中f(x)为权重，当词频过高时，权重不应过分增大，作者通过实验确定权重函数为： 

 <p align="center">
	<img src=./pictures/090805.png alt="Sample"  width="330">
	<p align="center">
		<em> </em>
	</p>
</p>

[如何记忆这个公式](<https://blog.csdn.net/coderTC/article/details/73864097>)

### 三、总结

1. 理论差异

两者最直观的区别在于，word2vec是基于**N-gram模型**理论，而GloVe是**统计模型**

2. 时间消耗

+ 不采用 negative sampling 的word2vec 速度非常快，但是：

  + 只告诉模型什么是有关的，却不告诉它什么是无关的，模型很难对无关的词进行惩罚从而提高自己的准确率
  + 当使用了negative sampling之后，为了将准确率提高到，word2vec训练就需要花较长时间
+ 相比于word2vec,因为golve更容易并行化，所以速度更快。
+ 使用sigmoid改进之后的word2vec训练时间也有所加快，两者的差异不再那么明显

3. 空间消耗

由于GloVe算法本身使用了全局信息，自然内存费的也就多一些，相比之下，word2vec在这方面节省了很多资源

4. 需求的语料规模

word2vec使用窗口里的单词作预测，需要较大语料。GloVe基于统计，在小语料也表现较好。