# Review of Neural Networks and Back Propagations

## Relevant Materials

- [CS 231n Notes](https://cs231n.github.io/neural-networks-1/)
- [CS 231n Notes](https://cs231n.github.io/optimization-2/)
- [Yes you should learning bp](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)

## 梯度数学基础

- 标量函数对标量输入的导数
- 标量函数对**向量**输入的偏导数，也是一个向量
- 矢量函数对矢量输入的偏导数，是一个对应的矩阵（雅可比矩阵）
- 激活函数的逐元素操作的导数

最终输出 s 对各层权重的偏导数可以被写成 **上游传递过来的误差信号**和**前向传播的变量**组成。

## 反向传播和计算图

### 建模

> https://openmlsys.github.io/chapter_computational_graph/components_of_computational_graph.html

计算图由基本数据结构张量（Tensor）和基本运算单元算子构成。在计算图中通常使用节点来表示算子，节点间的有向边（Directed Edge）来表示张量状态，同时也描述了计算间的依赖关系。

每一条边都对应了一个中间变量（或者参数），对应的，这条边的逆向边代表着反向传播的偏导数。

计算图中的**所有算子**都被严格依赖，并且整体的计算图是一个 DAG，不存在循环依赖的问题，这个性质也保证了计算图可以从最终的输出结果出发，逐步计算出各层变量的偏导数。

- 控制流算子（循环展开 & 分支）
- 数据流算子
- 张量操作算子
- 神经网络算子
