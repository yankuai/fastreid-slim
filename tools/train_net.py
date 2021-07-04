#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import sys
import os
import torch

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        ###############
        # checkpoint = torch.load('./logs/veri/sbs_R50-ibn/pruned/pruned_0.5.pth')
        # set cfg_channel
        # cfg.PRUNE.CFG_CHANNEL = checkpoint['cfg']

        # save state_dict file
        # data = {}
        # data["model"] = checkpoint['state_dict']
        # state_dict_filename = os.path.join(cfg.PRUNE.SAVE_DIR,"tmp.pth")
        # f = open(state_dict_filename,"wb")
        # torch.save(data, f)
        # f.close()

        model = DefaultTrainer.build_model(cfg)

        Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        #Checkpointer(model).load(state_dict_filename)

        res = DefaultTrainer.test(cfg, model)
        return res
    if args.refine:
        pruned_file_path = args.refine
        cfg.defrost()
        # 修改checkpoint输出位置为 OUTPUT_DIR\pruned_filename
        pruned_file_name = os.path.basename(pruned_file_path) # 文件名
        pruned_file_prename = os.path.splitext(pruned_file_name)[-2]   # [名称, 后缀]
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, pruned_file_prename)
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.makedirs(cfg.OUTPUT_DIR)
        # load checkpoint
        checkpoint = torch.load(pruned_file_path)
        # set cfg_channel
        cfg.PRUNE.CFG_CHANNEL = checkpoint['cfg']
        # save state_dict file
        data = {}
        data["model"] = checkpoint['state_dict']
        state_dict_filename = os.path.join(cfg.PRUNE.SAVE_DIR,"tmp.pth")
        f = open(state_dict_filename,"wb")
        torch.save(data, f)
        f.close()
        # set pretrain path
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.PRUNE.NEW_FEAT_DIM = True # use if refine densenet model 
        #cfg.MODEL.BACKBONE.PRETRAIN_PATH = state_dict_filename

    trainer = DefaultTrainer(cfg, {"sparsity_regularization": args.sr, "sparse_rate": args.s})

    if args.refine:
        trainer.checkpointer.load(state_dict_filename)
        # trainer.checkpointer.load('/home/my/fast-reid-slimming/fast-reid/logs/veri/sbs_R50-ibn/pruned_0.5/model_0029.pth')
        pass

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
