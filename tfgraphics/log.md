# Final: TensorFlow Graphics

项目概览https://github.com/tensorflow/graphics，中文翻译https://blog.csdn.net/qq_42233538/article/details/90201901

开启debug https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/debug_mode.md

## 2

### 项目思路

今天睡前目标：

- [x] 学tf estimator（可并行）。也看一下tensor board能不能展示一下模型（最好是运行过程）

[tensor board for mesh](https://www.tensorflow.org/graphics/tensorboard) [tf.estimator](https://tensorflow.google.cn/api_docs/python/tf/estimator/Estimator?hl=zh-cn) 可能有用的estimator[教程](https://www.cnblogs.com/marsggbo/p/11232897.html)

失败：好像没有找到合适的方法去改项目的执行路径，只能在分配的环境里面执行，不能在drive里面执行。不知道为什么。如果有空的话，可以从如何在drive中pip install下手去看看怎么运行。感觉上再改不太可能，就把模型copy到drive中算了。

找不到summary在哪，也不会打印。另外就是，总不能通过总是存模型来判断accuracy，这也太慢了吧。明天问问助教。

目前先改一下，~~多存checkpoint，多存点~~。把不必要的代码删除不用。开始看怎样修改模型合适。

发现了！进行test和checkpoint没有什么关系的。需要从evaluation的configuration部分进行更改，[API](https://tensorflow.google.cn/api_docs/python/tf/estimator/EvalSpec?hl=zh-cn)。

commit了。checkpoint和evaluation基本是同时发生的，所以evaluation的step基本是没用的。

跑一下

- [x] 写一下对输出的绘图分析程序（重要）

1. 首先evaluation的那些行全部替换掉
2. 1000个loss

> git log --pretty=oneline; git reset --soft 019d210de02415ecdafcb1c7ebf7f12b07e6c13c; git push origin master/main --force

- [ ] #### 看一下残差网络应该怎样魔改。因为size不一样啊…………（可并行）

resnet tensorflow[实现](https://github.com/raghakot/keras-resnet/blob/master/resnet.py) 可以看一下看一下。

简单改了一下加入残差的部分，还没有跑出来结果。后续看一下resnet怎么实现的，改的更复杂一点。可能的关注点：

- 残差中间应该封装什么网络？
- 残差是否应该和network in network结合起来？
- 如果要是修改原本的网络结构，需要对比跑一下。

引用这篇[文章](https://blog.csdn.net/qq_37555071/article/details/108258862?spm=1001.2014.3001.5502)，残差网络是可以加1\*1卷积操作的。

- [x] 先对整个图卷积模块加一下
- [x] 每一个图卷积层加一下

- [ ] #### 看一下network in network应该怎么加（重要）

- try1: 把num_filters改成了16，这个改变仅仅对图卷积起作用。不知道会不会有什么影响。

  影响，基本没有。

- try2:不改图卷积部分，改一下图卷积结束之后的模块。

  问题：要改成先concate之后再变16，还是先都16之后再concat

  - [x] 先concat，然后直接16，有效果的
  - [ ] 然后试一下平板上画的那个

- try3:改图卷积部分。

  问题：如何将3种128拼起来？

  - [x] 再加一层，拼起来之后多一层1*1卷积，loss降得非常非常快
  - [x] 在原本是64层那一层拼起来，输给128

  由于loss降得太快，需要降低learning rate.

  ~~调一下合适的learning rate。实在困姚明了。运行时已经存好。~~

- try4: 结合残差和network in network

上午问一下刘学长，对这个残差和inception的理解。问问看自己理解的对不对。

- [ ] 计算网络参数个数（必要）

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

Inception0: 多选几种kernel，最终合并几个kernel的计算结果。motivation是，对不同的大小范围进行感知。

Inception v1:先用1*1kernel降通道数，再进行多种kernel的计算。motivation是，先降维减少通道数量

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

## 3

开始了开始了

用OpenGL Assimp 读取off文件可视化即可。所以是两个任务：

~~opengl assimp环境配置以及文件读取和可视化，着色。官方[文档](https://assimp-docs.readthedocs.io/en/latest/about/index.html)，[learnOpenGL](https://learnopengl-cn.github.io/03%20Model%20Loading/01%20Assimp/)~~

- [x] 用linux系统下的geomview来可视化。

需要注意的大问题：win下生成的off文件不能在linux虚拟机上运行。所以需要用linux本身的文件，然后把内容co进去。直接copy是可以的。

安装运行：

```shell
sudo apt-get install geomview
geomview
```

- [x] 跑通github[实例代码](https://github.com/ozkanyumsak/mesh-subdivision)

他自己用的可视化工具好像是open inventor。open inventor好像用不成，读一下他写的类，看看能不能生成.obj文件. 代码结构看ipad上画的即可。

.off文件格式[参考](https://zhuanlan.zhihu.com/p/148859062)；

~~看懂其中的几种算法，然后改成自己的形式；~~

- [x] 写个脚本之类的东西，把程序发布出来，多跑一些模型写报告用。

- [x] 改好了用人体模型的代码，cat不能用，loop里面会出现inf
- [ ] 

## 报告

### 2

两个改法的原理要理解，写出来

然后画图，网络结构

**算出来参数数目**

把python绘图贴上去

### 3

算法伪代码

程序架构

运行效果贴图

## 答辩准备

需要准备一下3的脚本应该如何讲出来

