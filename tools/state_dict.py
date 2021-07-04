import torch

state_dict1 = torch.load('logs/veri/sbs_R50-ibn/model_final.pth')
state_dict2 = torch.load('logs/veri/sbs_R50-ibn/pruned/tmp.pth')

f1 = open('s0.txt',"w")
f1.write(str(state_dict1))
f2 = open('s1.txt',"w")
f2.write(str(state_dict2))
f1.close()
f2.close()