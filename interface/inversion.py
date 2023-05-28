from argparse import Namespace
import time
import torch
import torchvision.transforms as transforms
import dlib
import numpy as np
from PIL import Image

from pixel2style2pixel.utils.common import tensor2im
from pixel2style2pixel.models.psp import pSp
from pixel2style2pixel.scripts.align_all_parallel import align_face


class InversionModel:
    def __init__(self, checkpoint_path: str, dlib_path: str) -> None:
        self.dlib_path = dlib_path
        self.dlib_predictor = dlib.shape_predictor(dlib_path)

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
        opts["learn_in_w"] = False
        opts["output_size"] = 1024

        self.opts = Namespace(**opts)
        self.net = pSp(self.opts)
        self.net.eval()
        self.net.cuda()
        print("Model successfully loaded!")

    def run_alignment(self, image_path: str):
        aligned_image = align_face(filepath=image_path, predictor=self.dlib_predictor)
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image

    def inference(self, image_path: str):
        input_image = self.run_alignment(image_path)
        input_image = input_image.resize((256, 256))
        transformed_image = self.tranform_image(input_image)

        with torch.no_grad():
            tic = time.time()
            result_image, latents = self.net(
                transformed_image.unsqueeze(0).to("cuda").float(),
                return_latents=True,
                randomize_noise=False,
            )
            toc = time.time()
            print("Inference took {:.4f} seconds.".format(toc - tic))

        output_image = tensor2im(result_image[0])
        image = np.array(output_image.resize((256, 256)))
        res_image = Image.fromarray(image)
        return (
            res_image,
            {
                "w1": latents.cpu().detach().numpy(),
                "w1_initial": latents.cpu().detach().numpy(),
            },
        )
