import torch
from deprecated.helper import *
import argparse
from scipy.io import savemat

parser = argparse.ArgumentParser(description="PyTorch RateRNN training")
parser.add_argument("--path", default="", type=str, help="")
parser.add_argument("--mdl", default="", type=str, help="")

args = parser.parse_args()
loaded = torch.load(args.path + args.mdl + ".pth.tar")
matdict = py2mat(loaded)
savemat(args.path + args.mdl + ".mat", matdict)
