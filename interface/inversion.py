from argparse import Namespace
import time
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from huggingface_hub import snapshot_download

from pixel2style2pixel.datasets import augmentations
from pixel2style2pixel.utils.common import tensor2im, log_input_image
from pixel2style2pixel.models.psp import pSp


class InversionModel:
    def __init__(self, checkpoint_path: str, dlib_path: str) -> None:

        self.dlib_path = dlib_path
        self.tranform_image = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["checkpoint_path"] = checkpoint_path   
        opts['learn_in_w'] = False
        opts['output_size'] = 1024

  
        self.opts = Namespace(**opts)
        self.net = pSp(opts)
        self.net.eval()
        self.net.cuda()
        print('Model successfully loaded!')
