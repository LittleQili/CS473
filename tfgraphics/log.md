# Final: TensorFlow Graphics

项目概览https://github.com/tensorflow/graphics，中文翻译https://blog.csdn.net/qq_42233538/article/details/90201901

开启debug https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/g3doc/debug_mode.md

### 基础知识

贯穿始终的问题：减少训练的参数量。

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

