### 训练流程

在服务器安装miniconda

按照[INSTALL.md](https://github.com/yankuai/fast-reid/blob/master/docs/INSTALL.md)配置环境

按照[GETTING_STARTED.md](https://github.com/yankuai/fast-reid/blob/master/docs/GETTING_STARTED.md)下载和放置数据库，命令行跑代码

```
./tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml MODEL.DEVICE "cuda:0"
python tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml --num-gpus 4

# eval
python tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml --eval-only MODEL.WEIGHTS /path/to/checkpoint_file MODEL.DEVICE "cuda:0"

# help
./tools/train_net.py -h

# resume
./tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml --resume MODEL.DEVICE "cuda:0"

# 只加载模型state_dict，不恢复optimizer,epoch
./tools/train_net.py --config-file ./configs/VeRi/sbs_R50-deep.yml -sr --s 0.00001 MODEL.WEIGHTS ./logs/veri/sbs_R50-deep/model_0004.pth MODEL.DEVICE "cuda:0"
```

network slimming

```
# 稀疏化训练
./tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml -sr --s 0.00001 MODEL.DEVICE "cuda:0"

# 剪枝
./tools/prune_slimming.py --config-file ./configs/VeRi/sbs_R50-ibn.yml MODEL.WEIGHTS ./logs/veri/sbs_R50-ibn/model_final.pth MODEL.DEVICE "cuda:0"

# finetune 
./tools/train_net.py --config-file ./configs/VeRi/sbs_R50-ibn.yml --refine ./logs/veri/sbs_R50-ibn/pruned/pruned_0.3.pth MODEL.DEVICE "cuda:0"

# resnetdeep稀疏化训练
./tools/train_net.py --config-file ./configs/VeRi/sbs_R50-deep.yml -sr --s 0.00001 MODEL.DEVICE "cuda:0"
```

查看gpu使用情况

```
nvidia-smi
```

后台进程

```
tmux new -s mayan
tmux attach -t mayan
tmux kill-session -t mayan
```

杀死进程

```
# 查看进程
ps -ef
# 杀死进程
kill -s 9 88464		#88464是进程号
```

结果可视化

```
tensorboard --logdir /home/my/fast-reid-slimming/fast-reid/logs/veri/sbs_R50-deep
```

使用多个gpu，RuntimeError: Address already in use。

解决方案：设置不同的--dist-url[（fast-reid）多GPU训练出现RuntimeError: Address already in use解决 - 灰信网（软件开发博客聚合） (freesion.com)](https://www.freesion.com/article/77681373376/)

```
CUDA_VISIBLE_DEVICES='0,1' python ./tools/train_net.py --config-file ./configs/VeRi/sbs_R50-deep.yml -sr --s 0.00001 --num-gpus 2 --dist-url tcp://127.0.0.1:50001
 
CUDA_VISIBLE_DEVICES='0,1,2,3' python ./tools/train_net.py --config-file ./configs/VeRi/sbs_R50-deep.yml -sr --s 0.00001 --num-gpus 4 
```



### 下载数据库

从百度云直接下载到服务器方法参考 [跳过百度网盘客户端快速下载_夏佐的博客-CSDN博客](https://blog.csdn.net/qq_28125445/article/details/96435916?utm_medium=distribute.pc_relevant.none-task-blog-searchFromBaidu-4.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-searchFromBaidu-4.control)



### 踩坑一

报错

```
ModuleNotFoundError: No module named 'apex'
```

按照教程自己安装apex： https://blog.csdn.net/qq_38343151/article/details/107925586 第二个方法可行

但必须先启用cuda，服务器上面的安装在了 /usr/local/cuda，添加到自己的环境变量里，参考教程： https://blog.csdn.net/weixin_35683697/article/details/112011833?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.control

https://blog.csdn.net/qq_41111734/article/details/111057266  **如何在服务器上使用指定版本的cuda



### 踩坑二（读取配置文件cfg）

报错 

```
KeyError: 'Non-existent config key: SOLVER.MAX_ITER'
```

fast-reid-master/fastreid/config/defaults.py 中没有定义SOLVER.MAX_ITER

CfgNode实例的初始化参数默认 new_allowed=False，说明原node中不能添加新key

fast-reid-master/configs/VeRi/sbs_R50-ibn.yml 和 fast-reid-master/configs/VehicleID/bagtricks_R50-ibn.yml中都设置了SOLVER.MAX_ITER

解决方法：在fast-reid-master/fastreid/config/defaults.py 中定义缺少的SOLVER.MAX_ITER等等配置

![image-20210322120654881](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20210322120654881.png)



### 踩坑三(conda的channel配置)

##### 问题一

```
conda uninstall torchvision
```

报错 

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url 
```

<https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/repodata.json>

解决办法 删掉家目录下.condarc中的 - defaults

##### 问题二

```
conda install torchvision
```

报错

```
CondaHTTPError: HTTP 000 CONNECTION FAILED for url 
```

<https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-64/repodata.json>

解决办法 删掉家目录下.condarc中的 - defaults，镜像地址如果是https改成http。

##### 问题三

```
conda install torchvision
```

只能到0.2.1。所添加的镜像只包含低版本torchvision，应该添加额外的镜像，参考踩坑五。





### 踩坑四（预训练模型）

raise error

```
/export/home/lxy/.cache/torch/checkpoints/resnet50-19c8e357.pth is not found! Please check this path
```

下载好预训练模型

网址写在 fast-reid-master/fastreid/modeling/backbones/resnet.py

比如VeRi需要的模型是 https://download.pytorch.org/models/resnet50-19c8e357.pth

保存位置为 /home/my/checkpoint/resnet50-19c8e357.pth

修改 fast-reid-master/configs/Base-bagtricks.yml 文件中的

PRETRAIN_PATH: "/home/my/checkpoint/resnet50-19c8e357.pth"



### 踩坑五(安装合适版本的pytorch torchvision cuda)

报错

```
AssertionError: Torch not compiled with CUDA enabled
```

检查问题

```
import torch

print(torch.version.cuda)
```

可能之前下载的pytorch不是gpu版本。按照fastreid github安装指南安装pytorch造成的问题。

用pytorch官网命令行重新安装pytorch=1.6，torchvision=0.7，与fastreid github安装指南最大的不同是加上了cudatoolkit。

根据这个网站查看相应版本的安装指令 https://pytorch.org/get-started/previous-versions/

安装指令 

```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 tensorboard # 6006的cuda版本是10.2
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge  # 6059的cuda版本是11.1
```

(注意cuda版本，pytorch版本，torchvision版本三者相匹配。如果服务器root上安装的cuda与项目要求的pytorch版本不匹配，要自己安装新cuda在自己home中。)

##### 问题一

发现所有镜像中没有torchvision=0.7.0，添加新的镜像

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/

##### 问题二

pytorch=1.6.0下载速度很慢，conda install时会下载中断，可以先wget, -c是断点续传

```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0.tar.bz2 -c

wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/pytorch-1.8.0-py3.7_cuda11.1_cudnn8.0.5_0.tar.bz2 -c
```

再从本地安装

```
conda install --use-local pytorch-1.6.0-py3.7_cuda10.2.89_cudnn7.6.5_0.tar.bz2
```

##### **问题三**

pip install requirements太慢，换成conda

```
conda install --yes --file requirements
```

conda中没有opencv-python，可以先执行下面指令安装opencv

```
conda install -c https://conda.anaconda.org/menpo opencv
```



版本管理

git push不上去，git pull不下来（要设置rebase）

.gitignore忽略数据库

