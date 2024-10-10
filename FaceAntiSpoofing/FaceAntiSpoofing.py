import torch
import torchvision.transforms as transforms
from dassl.utils import  set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import argparse

import trainers.clip
import datasets.UniAttackData
from util.evaluator import FAS_Classification

from PIL import Image
import torch.nn.functional as F

__model__ = "/data/mahui/UniAttackData/output/CLIP-VL/CLIP@class/vit_b16/p1@UniAttack@UniAttack@UniAttack/seed1/CLIP@VL/checkpoint.pth"

def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.seed:
        set_random_seed(args.seed)

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.protocol:
        cfg.DATASET.PROTOCOL = args.protocol

    if args.protocol:
        cfg.DATASET.PREPROCESS = args.preprocess
        
    cfg.TEST.FINAL_MODEL = 'best_val'
    cfg.TEST.EVALUATOR = "FAS_Classification"
    
def extend_cfg(cfg,args):
    """ 
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER = CN()
    cfg.TRAINER.GPU = [int(s) for s in args.gpu_ids.split(',')]

    cfg.TRAINER.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.UPDATE = True

    cfg.TRAINER.CLIP = CN()
    cfg.TRAINER.CLIP.VERSION = args.version
    cfg.TRAINER.CLIP.PROMPT = args.prompt
    
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)
    if args.inference:
        cfg.inference = True
    else:
        cfg.inference = False
    cfg.freeze()

    return cfg

def main(args):
    cfg = setup_cfg(args)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    
    trainer = build_trainer(cfg)
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        
 ])
    img = transform(args.img)
    pre_input = {"frame":img.unsqueeze(0),
             "label":torch.tensor(0),
             "text":"none"}
    input,label = trainer.parse_batch_test(pre_input)
    output = trainer.model_inference(input)
    threshold = 1
    probabilities = torch.softmax(output, dim=1)
    print(probabilities.data.cpu().numpy()[0][1])
    return probabilities.data.cpu().numpy()[0][1] >= threshold
    
def faceAntiSpoofingByPath(path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, default=Image.open(path).convert('RGB').rotate(180))
    parser.add_argument("--gpu_ids", type=str, default="5")
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--preprocess", type=str, default="resize_crop_rotate_flip")
    parser.add_argument("--trainer", type=str, default="CLIP")
    parser.add_argument("--version", type=str, default="VL")
    parser.add_argument("--prompt", type=str, default="class")
    parser.add_argument("--model_dir", type=str, default="/data/mahui/UniAttackData/output//CLIP@class/vit_b16/p1@UniAttack@UniAttack@UniAttack/seed1/")
    parser.add_argument("--USE_CUDA", type=bool, default=True)
    parser.add_argument("--dataset_config_file", type=str, default="configs/datasets/UniAttackData.yaml")
    parser.add_argument("--config_file", type=str, default="configs/trainers/CLIP/vit_b16.yaml")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--protocol", type=str, default="p1@UniAttack@UniAttack@UniAttack", help="protocol")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--inference",default=True,action="store_true", help="inference mode")
    args = parser.parse_args()
    return main(args)
