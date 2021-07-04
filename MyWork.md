My work adds some models in */home/mayan/fast-reid/fastreid/modeling/backbones* and add some scripts in */home/mayan/fast-reid/tools*.

**Models**

The paper experimented with three backbones: resnet, resnetx2(resnet-v2), densenet.

Other backbones do not perform well.

|                           | 残差块                                                       | 头                           | 尾             |
| ------------------------- | ------------------------------------------------------------ | ---------------------------- | -------------- |
| resnet                    | conv(1,1)-bn-conv(3,3)-bn-conv(1,1)-bn<br />downsample:conv(1,1)-bn<br />[3,4,6,3] | conv(7,7)-bn-relu-maxpool(3) | 无             |
| resnetdeep                | bn-select-conv(1,1)-bn-conv(3,3)-bn-conv(1,1)<br />downsample:conv<br />[10,10,10] | conv(3,3)                    | bn-select-relu |
| resnet2                   | bn-select-conv(1,1)-bn-conv(3,3)-bn-conv(1,1)<br />downsample:conv<br />[3,4,6,3] | conv(7,7)-maxpool(3)         | bn-select-relu |
| resnet3                   | bn-select-conv(1,1)-bn-conv(3,3)-bn-conv(1,1)<br />downsample:conv<br />[3,4,6,3] | conv(7,7)-bn-relu-maxpool(3) | 无             |
| resnet4                   | bn-conv(1,1)-bn-conv(3,3)-bn-conv(1,1) <br />downsample:conv-bn<br />[3,4,6,3] | conv(7,7)-bn-relu-maxpool(3) | 无             |
| resnet5                   | bn-conv(1,1)-bn-conv(3,3)-bn-conv(1,1)<br />downsample:conv<br />[3,4,23,3] | conv(7,7)-bn-relu-maxpool(3) | 无             |
| resnetx                   | bn-select-conv(1,1)-bn-conv(3,3)-bn-conv(1,1)<br />downsample:conv<br />[10,10,10] | conv(3,3)                    | 无             |
| resnetx2<br />(resnet-v2) | bn-select-conv(1,1)-bn-conv(3,3)-bn-conv(1,1)<br />downsample:conv<br />[3,4,23,3] | conv(7,7)-bn-relu-maxpool(3) | 无             |
| densenet                  | dense block: bn-select-conv(1,1)-bn-conv(3,3)<br />transition block: bn-select-conv(1,1)-avgpool(2) | conv(7,7)-bn-relu-maxpool(3) | bn-select-relu |

**Scripts**

resprune: prune resnet.

res-v2prune: prune resnet-v2.

denseprune: prune densenet.

denseprune-it: prune densenet iterately. (need to debug)



flops_counter: count the FLOPs to process a (1,3,256,256) size input.

get_module_name: print a network's named modules.

parameter_counter: count trainable parameters in the backbone, aggregate part and head.

state_dict: print state_dict in str format.



train_net: script to start the train.

train_net_x2: redundant train_net.