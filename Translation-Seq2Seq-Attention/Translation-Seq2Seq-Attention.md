

[原文链接](<https://blog.csdn.net/abcgkj/article/details/95180232>)

一、统计机器翻译
--------

1990s-2010s，人们主要使用的是统计机器翻译。其核心思想是：从数据中学习概率模型。假设我们想把法语翻译为英语，即给定法语句子 x，寻找最佳的英语句子 y。我们可以用下图来描述： 

![](https://img-blog.csdnimg.cn/20190709114502580.png) 

然后，我们可以使用贝叶斯来把上式分解成两个部分，如下图所示： 

![](https://img-blog.csdnimg.cn/20190709114517196.png) 

其中，P(x|y): 可以视为翻译模型。模型从并行数据中学习，单词或句子应该如何被翻译。P(y)：可以被视为语言模型。在本例子中，从英语数据中学习如何写好英语 (流利)。之前的学习中，我们已经介绍过语言模型，这里不再赘述。因此，如何得到翻译模型 P(x|y) 是重点。下面我们分步来介绍传统的机器翻译是怎样实现的：

第一：我们需要一个大的平行语料库据（例如：翻译成法语 / 英语的句子对)。下图是世界上第一个平行语料库：罗塞塔石碑： 

<p align="center">
	<img src=./pictures/090313.png alt="Sample"  width="450">
	<p align="center">
		<em> </em>
	</p>
</p>




第二：对齐（翻译句子中特定词语之间的对应关系）。即法语句子 x 与英语句子 y 之间的单词级对应。对齐时，原文中可能有部分词语没有对应的译文，如下图所示： 

<p align="center">
	<img src=./pictures/090314.png alt="Sample"  width="550">
	<p align="center">
		<em> </em>
	</p>
</p>




对齐可以是多对一的，如下图所示： 

<p align="center">
	<img src=./pictures/090315.png alt="Sample"  width="550">
	<p align="center">
		<em> </em>
	</p>
</p> 


对齐可以是一对多的，如下图所示： 

<p align="center">
	<img src=./pictures/090316.png alt="Sample"  width="550">
	<p align="center">
		<em> </em>
	</p>
</p>




当然，对齐可以是多对多 (短语级)，如下图所示： 

<p align="center">
	<img src=./pictures/090317.png alt="Sample"  width="550">
	<p align="center">
		<em> </em>
	</p>
</p>




对齐之后，原文中每个单词都有多个备选单词，导致了许多短语的组合方式，如下图所示： 

<p align="center">
	<img src=./pictures/090318.png alt="Sample"  width="650">
	<p align="center">
		<em> </em>
	</p>
</p>


第三：解码，即使用 heuristic search 算法搜索最佳翻译，丢弃概率过低的假设，如下图所示： 

<p align="center">
	<img src=./pictures/090319.png alt="Sample"  width="650">
	<p align="center">
		<em> </em>
	</p>
</p> 


以上所述，这还只是传统机器翻译系统的冰山一角，有许多细节没有涉及到，还需要大量的特征工程和人力维护，总之是非常复杂的系统。其中每个环节都是独立不同的机器学习问题。

而深度学习则提供了一个统一的模型，一个统一的最终目标函数。在优化目标函数的过程中，得到一个 end to end 的完整的 joint 模型。传统机器翻译系统与深度学习是截然相反的，对齐模型、词序模型、语言模型…… 一堆独立的模型无法联合训练。接下来我们来介绍神经机器翻译模型。

二、神经网络机器翻译
-----------------

2014 年，Neural Mechine Translation 出现了。神经机器翻译 (NMT) 是一种使用单一神经网络进行翻译的机器翻译方法。神经网络结构称为 sequence-to-sequence(又名 seq2seq)。举个例子，我们的翻译任务定义为：

输入：一个法语句子（Source sentence）：il a m’ entarté。( entarté 是法语中“用派打某人”的意思)

输出：一个英语句子（Target sentence）：he hit me with a pie。（他用一个派打我）

+ 全局的观察下神经网络机器翻译的seq2seq模型

<p align="center">
	<img src=./pictures/090300.PNG alt="Sample"  width="800">
	<p align="center">
		<em> </em>
	</p>
</p>

+ 编码器encoder

Encoder 负责将输入的原文本编码成一个向量（context），该向量是原文本的一个表征，包含了文本背景。为解码器提供初始隐藏状态。如下图所示： 

<p align="center">
	<img src=./pictures/090301.png alt="Sample"  width="400">
	<p align="center">
		<em> </em>
	</p>
</p>

+ 解码器decoder

Decoder 是一种以编码为条件生成目标句的语言模型，即使用 Encoder 的最终状态和作为 Decoder 的初始隐状态。除此之外，Decoder 的每个cell的输入还来源于前一个时刻的隐藏层和前一个预测结果。如下图所示，这是在测试时的 Decoder，下文会介绍如何训练： 

<p align="center">
	<img src=./pictures/090302.png alt="Sample"  width="500">
	<p align="center">
		<em> </em>
	</p>
</p> 

注意: 这里的 Encoder 和 Decoder，可以是普通的 RNN，也可以是 LSTM、GRU 或者是 Bi-LSTM 等等，当然也可以是 CNN。层数也可以是多层的。当然，我们不仅可以用同一种神经网络实现编码器和解码器，也可以用不同的网络，如编码器基于 CNN，解码器基于 RNN。

+ 如何训练NMT

那么，我们如何训练一个 NMT 系统呢？和传统的机器翻译系统一样，首先我们需要一个大的平行语料库，然后我们对 Source sentence 进行编码，然后使用 Encoder 的最终隐藏状态作为 Decoder 的初始隐藏状态进行预测 y1，y2……y7，然后用 Decoder 中输入的 Target sentence 和预测出来的 y 计算 loss（例如交叉熵损失函数），最后使用反向传播来更新模型中的参数。如下图所示： 
<p align="center">
	<img src=./pictures/090303.png alt="Sample"  width="800">
	<p align="center">
		<em> </em>
	</p>
</p>  

图片中，在 Decoder 里我们通过 argmax 来选择每一步的单词，然后将生成的单词又输入到cell中生成下一步的单词，这种方式我们称之为贪婪解码。

<p align="center">
	<img src=./pictures/090302.png alt="Sample"  width="500">
	<p align="center">
		<em> </em>
	</p>
</p> 

但是，这使得在生成前面的单词后无法回溯。那么我们如何解决这个问题呢？不难想到，我们可以计算所有可能的情况，但很明显这样的计算代价是非常大的。因此，我们采用来一种新的方式——Beam search decoding集束搜索。

+ Beam search decoding 集束搜索解码

Beam search decoding 的核心思想是：在解码器的每个步骤中，跟踪 k 个最可能的部分 (我们称之为假设，其实就是选择k个得分最高的分支，其它分支就砍掉了)，其中 k 我们称之为 beam size（大约是 5 到 10）。

<p align="center">
	<img src=./pictures/090304.png alt="Sample"  width="800">
	<p align="center">
		<em> </em>
	</p>
</p>  

Beam search 虽然不能保证我们找到最优解，但比穷举搜索更有效。通常我们在进行 t 次个词后可以停止 Beam search。且通常情况下，越长的假设分支得分会越低，这是不合理的，最后我们需要在计算总得分后除以 t，如下图所示 ：
<p align="center">
	<img src=./pictures/090305.png alt="Sample"  width="350">
	<p align="center">
		<em> </em>
	</p>
</p>  

经过上文，我们很容易理解所谓的 seq2seq，它其实是一个 Encoder 和一个 Decoder，Encoder 用来编码 source sentence，Decoder 使用 Encoder 的最终隐藏状态作为其的初始隐藏状态，充当为一个语言模型来预测单词，然后根据 Target sentence 计算 loss 来更新模型。

那么，我们可以单独训练 Encoder 和 Decoder 吗？答案是可以的，但是同步训练的好处是，我们可以同时优化模型中的参数。当然，如果我们分开训练，也可以使用一个预训练好的语言模型作为 Decoder 的初始状态，然后针对你的任务进行 fine tune。此外，对于模型中需要的 word embeding，我们可以使用该任务的语料库得到，也可以使用现成的 word embeding，然后进行 fine tune。

三、NMT vs SMT
------------

与 SMT 相比，NMT 有很多优点:

*   更好的性能：更流利、更好地利用上下文、更好地使用短语相似性
*   单个神经网络端到端优化：没有需要单独优化的子组件
*   需要更少的人力工作：没有特征工程、所有语言对的方法相同

当然，也有一些缺点：

与 SMT 基于规则的方式相比，NMT 的可解释性较差：难以调试。例如：当我们使用 NMT 模型进行翻译时，如果发现翻译错误，我们很难找到是模型中的哪个神经元或者参数出了问题。

四、如何评估机器翻译模型
------------

BLEU（Bilingual Evaluation Understudy）是一个人们普遍运用的 MT 模型评价标注。BLEU 将机器翻译与一个或多个人工翻译进行比较，并根据以下条件计算相似度评分:

*   n-gram 精度 (通常为 1、2、3 和 4-grams)
*   此外，系统翻译太短也会受到惩罚

虽然 BLEU 很有用，但也有一些缺点。例如：一个句子可以有很多种翻译方式，因此一个好的翻译可能会得到一个较差的 BLEU 评分，因为它与人工翻译的 n-grams 重叠度较低。但是作为自动评价机器翻译系统好坏的方法，我们仍不得不暂时使用 BLEU 作为机器翻译系统的评价标准。

五、注意力机制
-------

### 1. 动机

上文中，我们提到的 seq2seq 模型中，很重要的一个步骤是，把 Encoder 的最终唯一的隐藏状态当作 Decoder 的初始隐藏，这样做造成Decoder产生瓶颈问题，即Decoder获得的隐状态输入所包含的信息太少。源语句的编码必须捕获关于源语句的所有信息，才可以使模型很好的工作，因此注意力机制的提出就可以很好的解决这个瓶颈。

注意力机制的核心思想是：在解码器的每个步骤上，使用与“编码器的一个直接连接”来注意源序列的特定部分。

简单来说，注意力机制为解码器网络提供了在每个解码步骤查看整个输入序列的功能，然后解码器可以在任何时间点决定哪些输入单词是重要的。接下来我们通过公式来从本质上理解一下 Attention 机制。

### 2. 公式理解

<p align="left">
	<img src=./pictures/090306.png alt="Sample"  width="600">
	<p align="center">
		<em> </em>
	</p>
</p>  


首先，来做一下定义：编码器的隐藏状态为: h_1,……,h_N。在时间步 t，解码器的隐藏状态为：s_t。

用t时刻decoder的隐状态s_t点乘encoder中的N个隐状态得到N个标量，就得到 t 时间步下的 attention 分数e^t：

<p align="center">
	<img src=./pictures/090307.png alt="Sample"  width="450">
	<p align="center">
		<em> </em>
	</p>
</p> 

并使用 softmax 函数进行归一化处理得到α^t：

<p align="center">
	<img src=./pictures/090308.png alt="Sample"  width="650">
	<p align="center">
		<em> </em>
	</p>
</p> 

然后，我们用向量α^t 的每个值作为权值，与编码器隐藏层状态进行加权求和，得到 attention 的输出 a_t（有时也称之为 context vector），如下图所示： 

<p align="center">
	<img src=./pictures/090309.png alt="Sample"  width="700">
	<p align="center">
		<em> </em>
	</p>
</p> 

最后，我们将注意力的输出与解码器隐状态连接起来，并按照非注意 seq2seq 模型进行处理，或者直接将注意力的输出喂给下一个cell，如下图所示： 

<p align="center">
	<img src=./pictures/0903010.png alt="Sample"  width="700">
	<p align="center">
		<em> </em>
	</p>
</p> 

<p align="center">
	<img src=./pictures/0903011.png alt="Sample"  width="700">
	<p align="center">
		<em> </em>
	</p>
</p> 

以上就是一般情况下 attention 机制的公式描述。

+ 广义attention定义

接下来我们给出更加广义的 attention 定义：

给定一组向量值（encode过程中所有输出的隐状态，一个隐状态就是一个向量）和一个向量查询querry（decode过程中t时刻的输出的隐状态），

注意力机制是一种根据查询计算值的加权和的技术（我们的注意力就体现在我们的查询中）。因此有时，我们也简单称之为查询处理值。例如，在 seq2seq + attention 模型中，每个解码器隐藏状态 (查询) 都关注所有编码器隐藏状态(值)。因此注意力机制中的加权和是值中包含的信息的选择性汇总，查询在其中确定要关注哪些值。

+ attention 机制中计算attention分数的几种常见形式：

  假设我们拥有一些值 h_1,……,h_N，维度为 d_1 和一个查询 s，维度为 d_2，且 d_1 = d_2，则：
  + Basic dot-product attention(和上文介绍的 attention 一致)： 
    
    ![](https://img-blog.csdnimg.cn/20190709115241118.png)
    
  + Multiplicative attention：这里的 W 是一个维度为 d_2 * d_1 的权重矩阵，也是我们需要更新的参数之一
    
    ![](https://img-blog.csdnimg.cn/20190709115253571.png)
  
  
  
  + Additive attention：这里的 W_1 的维度是 d_3 *d_1，W_2 的维度是 d_3 *d_2，v 是一个维度为 d_3 的权重向量，d_3 也是模型中的一个超参数 
    
    ![](https://img-blog.csdnimg.cn/20190709115302580.png)
  
  

小结
--

本节课我们首先回顾了传统的统计机器翻译（SMT），接着讲述了神经机器翻译（NMT）——seq2seq，最后利用 Attention 机制 seq2seq 模型进行了改进。当然，此时的 seq2seq 模型仍存在一些缺点，比如 OOV（Out-of-Vocabulary）问题，可以通过指针或者复制机制来处理，又比如大规模输出词汇处理的问题、训练和测试数据之间的域不匹配（例如：训练数据采用正式的文本，而测试数据采用聊天的非正式文本）等等。