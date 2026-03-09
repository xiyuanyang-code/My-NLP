# NLP-1 Word Vector

## History

- 图灵机 & 图灵测试
- 有限自动机 & 正则表达式
    - 非确定有限状态自动机
    - 博客：https://xiyuanyang-code.github.io/posts/Automaton-NFA/
- 基于统计概率的经验主义：离散马尔科夫过程的概率模型
    - 噪声信道模型
    - Shannon & 信息论 & 熵
- **基于符号规则的理性主义**
    - 结构化的语义表示
    - 预定义规则匹配

在 20 世纪 50 年代，受限于计算机硬件的能力贫瘠，基于符号规则的理性主义在文本智能问答、机器翻译等任务上表现更加优异，但是在日益复杂的实际任务中规则系统的局限性开始显现。

Data Driven: Every time I fire a linguist, the performance of the speech recognizer goes up.

- **统计学习算法**：
    - 决策树 & SVM 等方法引入
    - 概率图模型
    - 基于**示例**的机器翻译方法（语料库命中和修改）

- **基于神经网络的方法**

## Representing Words

### Signifier and Signified

索绪尔（Ferdinand de Saussure）结构主义语言学

1. 能指 (Signifier) —— 符号的形式：声音、感知、符号的物质层面。
2. 所指 (Signified) —— 符号的概念：心里的意向和概念。

能指和所指之间连接的构建是人为制定的。

### WordNet

A thesaurus containing lists of synonym sets and hypernyms

缺点：

- 词表需要实时更新，对于未知语义的词需要实时更新
    - 主观性
    - 人力成本高
- 无法有效的捕捉**词和词之间的相关性**


### Independent Words & Vectors

- 独热编码的向量
    - 缺点：词向量之间相互正交，缺乏语义联系
    - 仅受限在**有限语料库**的情况下，在庞大的词表下表示效率低下。
- Vectors from annotated discrete properties
    - 语法信息、语义信息
    - **You shall know a word by the company it keeps** (上下文很重要！)

### Word2Vec

A framework for learning word vectors.

- 每一个固定表示的字符串（单词，例如使用 BPE 进行分词可以被表示成一个**高维空间中的高维向量**。
- 基本假设：**上下文相似的词，其语义也相似**。
- 训练过程：根据上下文相关的文本进行分词训练，根据相似度点击进行概率计算，然后反向传播更新向量。

![`word2vec`](../image/word2vec.png)

Now computing $P(w_{t+j}|w_t)$:

Several maths, see handmade slides...

- 根据向量相似度进行概率建模（根据选中的词生成上下文）
- 最终利用负对数似然的损失函数进行最小化，得到优化目标
    - 缺点：全局的 softmax 计算过于复杂，尤其是词表变的非常大的时候。
- 参数：核心的 U 和 V 矩阵（embeddings 本身就是参数）

#### Word2Vec Families Details

- Skip-Grams
- Continuous Bag of Words (CBOW)

两者的核心 forward 架构相同，单个词的概率建模公式 $P(w_{t+j}|w_t)$ 也相同，对单个词的损失函数（负对数）也相同。区别在于具体训练的模式和总的损失函数存在差异：

- CBOW 的核心在于**根据上下文的词预测当前词**，将上下文向量求和并取平均。对应的，梯度回传的时候更新出来的梯度也会平均分给每一个词向量。

$$
J_{CBOW} = -\log P(w_t \mid w_{t-C}, \dots, w_{t+C})
$$

- Skip-Grams 的核心在于**根据当前次预测上下文的词**，并且认为这些上下文是独立的。

$$
J_{Skip-gram} = -\sum_{-C \le j \le C, j \ne 0} \log P(w_{t+j} \mid w_t)
$$

> 但是具体涉及到概率的计算和概率的反向传播过程，本质上没有什么区别。


#### SkipGram

核心：利用当前词来进行上下文预测

#### CBOW 模型

核心：利用上下文进行当前词的预测（类似于 BERT）

![CBOW](../image/cbow.png)

简化情况：上下文只包含了一个词的 CBOW 模型：

- 输入层从输入向量 x 进入到隐藏层 h，因为 x 是独热向量，因此输出的隐藏层向量 h 就是第一个 U 矩阵的参数。（向量被记为 v，即输入层到隐藏层的权重中）
- 输出矩阵 V 实现从隐藏层向输出概率的映射（向量被记为 u，即隐藏层到输出层的权重中）
- v 代表中心词向量（输入向量），u 代表上下文向量（输出向量）
- u^T v 最终输出的是概率分布（单点概率）

> **这里的向量内积**可以从相似度的角度解释，也可以从最终 softmax 输出概率分布的角度解释。
- 最终的 embedding 可以使用 **输入向量和输出向量的加权组合**（或者只保留输入向量）

> 如果 CBOW 是存在更多的上下文，则对输入的独特向量取平均，即输入向量 v 取平均。（但是只做一次训练）。但是 Skip-Gram 可以通过滑动窗口进行不同词的训练（在训练过程中中心词和上下文词的距离关系不存在差异，但是在真实的训练中可以使用**随机化窗口长度**的方法来提升离的近的词语的训练次数）

矩阵乘法的详细推导见纸质笔记。

### 词向量的优化策略

#### Hierarchical Softmax

- 将多分类问题转化为多个二分类问题（决策树）
- 构造最优二叉树：哈夫曼树
    - 基于哈夫曼树的哈夫曼编码具有最短的平均码长

哈夫曼树的计算次数是 log 级别的，这样可以将一个 softmax 的多分类问题转化为等价的二分类问题。哈夫曼树的构建可以直接利用训练集中不同单词出现的频率来进行构建。

#### Negative Sampling

Observe：除去正确单词之外，其余无关的词的概率层输出几乎都是 0，计算非常浪费。
Optimize: 优化 Softmax 的计算次数：
    - 最大化正确单词上的输出
    - **噪声分布**：随机取噪声词，最小化噪声单词的输出

噪声分布是一个根据频率而决定的动态分布：

$$
P(w_i|C) = \frac{\alpha_i^{\frac{3}{4}}}{\sum_j \alpha_j^{\frac{3}{4}}}
$$
    

### More

- **一词多义**：很多词的意思是在**上下文中所确定的**，而简单的词向量只是均匀的混合了这些意思。
- Context-Dependent Word Vectors for pre-training models. 在预训练模型中，同一个词在不同的句子中会有不同的向量表示。模型不再为每个词维护一个固定的查找表（Lookup Table），而是通过一个**深层网络（如 LSTM 或 Transformer）**根据该词周围的邻居实时“计算”出它的向量。