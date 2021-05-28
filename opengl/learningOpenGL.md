# OpenGL 学习

### 环境配置

> 参考[教程](https://learnopengl-cn.github.io/01%20Getting%20started/01%20OpenGL/)

直接采用[env_sync](.\env_sync)文件夹下的环境配置Visual Studio:

- 首先将编译器设置为x64（lib的生成环境是x64，以后有需求的话可以配置）
- 配置项目属性：
  - VC++目录-包含目录-include
  - VC++目录-库目录-lib
  - 链接器-输入-附加依赖项-添加glfw3.lib
- 将src里面的glad.c内容添加到工程源文件中。建议新建源文件，或者copy到：项目文件夹/项目名称文件夹下。

> 好像不太适合用linux环境进行编程。先用自己的电脑做吧