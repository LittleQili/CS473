# Final: TensorFlow Graphics

项目概览https://github.com/tensorflow/graphics，中文翻译https://blog.csdn.net/qq_42233538/article/details/90201901

开启debug https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/debug_mode.md

### 项目思路

今天睡前目标：

- [x] 学tf estimator（可并行）。也看一下tensor board能不能展示一下模型（最好是运行过程）

[tensor board for mesh](https://www.tensorflow.org/graphics/tensorboard) [tf.estimator](https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator?hl=zh-cn) 可能有用的estimator[教程](https://www.cnblogs.com/marsggbo/p/11232897.html)

失败：好像没有找到合适的方法去改项目的执行路径，只能在分配的环境里面执行，不能在drive里面执行。不知道为什么。如果有空的话，可以从如何在drive中pip install下手去看看怎么运行。感觉上再改不太可能，就把模型copy到drive中算了。

找不到summary在哪，也不会打印。另外就是，总不能通过总是存模型来判断accuracy，这也太慢了吧。明天问问助教。

目前先改一下，~~多存checkpoint，多存点~~。把不必要的代码删除不用。开始看怎样修改模型合适。

发现了！进行test和checkpoint没有什么关系的。需要从evaluation的configuration部分进行更改，[API](https://tensorflow.google.cn/api_docs/python/tf/estimator/EvalSpec?hl=zh-cn)。

commit了。checkpoint和evaluation基本是同时发生的，所以evaluation的step基本是没用的。

- [ ] 看一下残差网络应该怎样魔改。因为size不一样啊…………（可并行）

### 项目Toturial

3D mesh不太好处理。mesh可以被当作图来进行处理。图卷积？

3D mesh的对角线需要非零，需要weighted adjadency matrix.矩阵的定义:

```
A[i, j] = w[i,j] if vertex i and vertex j share an edge,
A[i, i] = w[i,i] for each vertex i,
A[i, j] = 0 otherwise.
where, w[i, j] = 1/(degree(vertex i)), and sum(j)(w[i,j]) = 1
```

每个mesh的基本信息：

*   'num_vertices', V: Number of vertices in each mesh.
*   'num_triangles', T: Number of triangles in each mesh.
*   'vertices': A [V, 3] float tensor of vertex positions.
*   'triangles': A [T, 3] integer tensor of vertex indices for each triangle.
*   'labels': A [V] integer tensor with segmentation class label for each
    vertex.

python with语法（[ref](https://blog.csdn.net/jiaoyangwm/article/details/79243756)）：就是在语句块内创立一个上下文环境。with后面加一个context manager对象管理器，在with语句块执行之前执行`enter()`执行后执行`exit()`.with as: as后面语句被`enter()`返回值赋值。

mesh encoder(vertices to C-dimensional logits, C is number of parts)

TensorFlow接口知识

[Conv1D](https://tensorflow.google.cn/api_docs/python/tf/keras/layers/Conv1D?hl=zh-cn)

```python
tf.keras.layers.Conv1D(
    filters, kernel_size, strides=1, padding='valid',
    data_format='channels_last', dilation_rate=1, groups=1,
    activation=None, use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```

### 基础知识

贯穿始终的问题：减少训练的参数量；让梯度有效下降。

#### Convolution

主要是计算。对于Conv2D来说，输入有几个通道，一个卷积kernel就得有几个平面；有几个卷积kernel，就有几个输出通道。

#### LeNet

文章https://blog.csdn.net/qq_37555071/article/details/107629340?spm=1001.2014.3001.5502 

文章末尾提供了卷积操作的基本计算公式。讲的满详细的。本网络主要就是conv的基本操作，+pooling以及最终的拍平。

#### AlexNet

https://zhuanlan.zhihu.com/p/116197079

相比于上一个网络，激活函数换成了ReLU，增加了Dropout层。

？LRN用于ReLU后面的一层。

#### VGG

https://zhuanlan.zhihu.com/p/116900199

主要特征是，卷积的kernel大小固定，变小。多个小卷积核能够相当于一个大卷积核的作用，并且引入更多的非线性性。有VGG-16和VGG-19。

#### GoogLeNet

[这篇文章](https://blog.csdn.net/qq_37555071/article/details/107835402?spm=1001.2014.3001.5502)讲Inception 讲的特别好。

这里面的concatenate指的是通道的合并。

Inception0: 多选几种kernel，最终合并几个kernel的计算结果。

Inception v1:先用1*1kernel降通道数，再进行多种kernel的计算。

Inception v2:多种kernel中的大kernel用多层小kernel来代替

Inception v3:大kernel非对称拆分。例如：n\*n-> 1\*n + n\*1

Bottleneck: 先用1\*1kernel降通道数，再正常卷积，再1\*1kernel升通道数。

具体讲GoogleNet的参数计算的[文章](https://blog.csdn.net/qq_37555071/article/details/108214680)

？**Xception**

#### ResNet

这样的结构

![image-20210605221409421](F:\term6\GPU\CS473\tfgraphics\log.assets\image-20210605221409421.png)

前向传播解决问题：网络过深带来的模型退化的问题。因为当一定层数的网络有效时，后面x值基本都稳定变化，Fx就趋近于0。

> 当浅层的输出已经足够成熟，让深层网络后面的层能够实现恒等映射的作用（即让后面的层从恒等映射的路径继续传递）

反向传播：引入残差，对输出的变化更加敏感。可以说一定程度上解决了梯度消散的问题。

[这篇博客](https://blog.csdn.net/qq_37555071/article/details/108258862?spm=1001.2014.3001.5502)用中文解释的比较清楚。

#### DenseNet

如果说ResNet是<img src="F:\term6\GPU\CS473\tfgraphics\log.assets\image-20210605223859665.png" alt="image-20210605223859665" style="zoom:50%;" />

那么DenseNet就是<img src="F:\term6\GPU\CS473\tfgraphics\log.assets\image-20210605223925292.png" alt="image-20210605223925292" style="zoom:50%;" />

其中，l表示layer,H可以表示BN, ReLU, Conv, Pooling之类的组合。

但是这里的concatenate方式和GoogleNet一样是通道的组合。所以需要采用前面说的Bottleneck进行参数减少的操作。

还是[这篇博客](https://blog.csdn.net/qq_37555071/article/details/108377880?spm=1001.2014.3001.5502)讲得好。

