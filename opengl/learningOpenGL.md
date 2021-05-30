# OpenGL 学习

### 环境配置

> 参考[教程](https://learnopengl-cn.github.io/01%20Getting%20started/01%20OpenGL/)

直接采用[env_sync](.\env_sync)文件夹下的环境配置Visual Studio:

- 首先将编译器设置为x64（lib的生成环境是x64，以后有需求的话可以配置）
- 配置项目属性：
  - VC++目录-包含目录-include
  - VC++目录-库目录-lib
  - 链接器-输入-附加依赖项-添加glfw3.lib glut.lib glut32.lib
- 将src里面的glad.c内容添加到工程源文件中。建议新建源文件，或者copy到：项目文件夹/项目名称文件夹下。

> 好像不太适合用linux环境进行编程。先用自己的电脑做吧
>
> F:\term6\GPU\CS473\opengl\env_sync\include
>
> F:\term6\GPU\CS473\opengl\env_sync\lib
>
> F:\term6\GPU\CS473\opengl\env_sync\src

https://blog.csdn.net/u013295276/article/details/78268601 作业链接

https://learnopengl.com/In-Practice/Debugging 调试教程



用gl库跑通的一个代码：https://github.com/SahibYar/Voronoi-Fortune-Algorithm/tree/master/Voronoi_cpp

配置环境教程，x86环境编译，添加一个链接库就行了：https://blog.csdn.net/weixin_41962350/article/details/109345558



想借用的教程 https://nullprogram.com/blog/2014/06/01/

CUDA https://blog.csdn.net/weixin_33872566/article/details/86034159

https://antongerdelan.net/opengl/  https://antongerdelan.net/opengl/hellotriangle.html

配置glew教程 https://blog.csdn.net/sunmenmian/article/details/88594161 **opengl32.lib**;glfw3.lib;glew32sd.lib sys32和syswow 64都记得复制一份

