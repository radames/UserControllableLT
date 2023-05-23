import os
from argparse import Namespace
import numpy as np
import torch

from models.StyleGANControler import StyleGANControler


class Model:
    def __init__(
        self, checkpoint_path, truncation=0.5, use_average_code_as_input=False
    ):
        self.truncation = truncation
        self.use_average_code_as_input = use_average_code_as_input
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        opts = ckpt["opts"]
        opts["checkpoint_path"] = checkpoint_path
        self.opts = Namespace(**ckpt["opts"])
        self.net = StyleGANControler(self.opts)
        self.net.eval()
        self.net.cuda()
        self.target_layers = [0, 1, 2, 3, 4, 5]

    def random_sample(self):
        z1 = torch.randn(1, 512).to("cuda")
        x1, w1, f1 = self.net.decoder(
            [z1],
            input_is_latent=False,
            randomize_noise=False,
            return_feature_map=True,
            return_latents=True,
            truncation=self.truncation,
            truncation_latent=self.net.latent_avg[0],
        )
        w1_initial = w1.clone()
        x1 = self.net.face_pool(x1)
        image = (
            ((x1.detach()[0].permute(1, 2, 0) + 1.0) * 127.5).cpu().numpy()[:, :, ::-1]
        )
        return (
            image,
            {
                "w1": w1.cpu().detach().numpy(),
                "w1_initial": w1_initial.cpu().detach().numpy(),
            },
        )  # return latent vector along with the image

    def latents_to_tensor(self, latents):
        w1 = latents["w1"]
        w1_initial = latents["w1_initial"]

        w1 = torch.tensor(w1).to("cuda")
        w1_initial = torch.tensor(w1_initial).to("cuda")

        x1, w1, f1 = self.net.decoder(
            [w1],
            input_is_latent=True,
            randomize_noise=False,
            return_feature_map=True,
            return_latents=True,
        )
        x1, w1_initial, f1 = self.net.decoder(
            [w1_initial],
            input_is_latent=True,
            randomize_noise=False,
            return_feature_map=True,
            return_latents=True,
        )

        return (w1, w1_initial, f1)

    def transform(
        self,
        latents,
        dz,
        dxy,
        sxsy=[0, 0],
        stop_points=[],
        zoom_in=False,
        zoom_out=False,
    ):
        w1, w1_initial, f1 = self.latents_to_tensor(latents)
        w1 = w1_initial.clone()

        dxyz = np.array([dxy[0], dxy[1], dz], dtype=np.float32)
        dxy_norm = np.linalg.norm(dxyz[:2], ord=2)
        dxyz[:2] = dxyz[:2] / dxy_norm
        vec_num = dxy_norm / 10

        x = torch.from_numpy(np.array([[dxyz]], dtype=np.float32)).cuda()
        f1 = torch.nn.functional.interpolate(f1, (256, 256))
        y = f1[:, :, sxsy[1], sxsy[0]].unsqueeze(0)

        if len(stop_points) > 0:
            x = torch.cat(
                [x, torch.zeros(x.shape[0], len(stop_points), x.shape[2]).cuda()], dim=1
            )
            tmp = []
            for sp in stop_points:
                tmp.append(f1[:, :, sp[1], sp[0]].unsqueeze(1))
            y = torch.cat([y, torch.cat(tmp, dim=1)], dim=1)

        if not self.use_average_code_as_input:
            w_hat = self.net.encoder(
                w1[:, self.target_layers].detach(),
                x.detach(),
                y.detach(),
                alpha=vec_num,
            )
            w1 = w1.clone()
            w1[:, self.target_layers] = w_hat
        else:
            w_hat = self.net.encoder(
                self.net.latent_avg.unsqueeze(0)[:, self.target_layers].detach(),
                x.detach(),
                y.detach(),
                alpha=vec_num,
            )
            w1 = w1.clone()
            w1[:, self.target_layers] = (
                w1.clone()[:, self.target_layers]
                + w_hat
                - self.net.latent_avg.unsqueeze(0)[:, self.target_layers]
            )

        x1, _ = self.net.decoder([w1], input_is_latent=True, randomize_noise=False)

        x1 = self.net.face_pool(x1)
        result = (
            ((x1.detach()[0].permute(1, 2, 0) + 1.0) * 127.5).cpu().numpy()[:, :, ::-1]
        )
        return (
            result,
            {
                "w1": w1.cpu().detach().numpy(),
                "w1_initial": w1_initial.cpu().detach().numpy(),
            },
        )

    def change_style(self, latents):
        w1, w1_initial, f1 = self.latents_to_tensor(latents)
        w1 = w1_initial.clone()

        z1 = torch.randn(1, 512).to("cuda")
        x1, w2 = self.net.decoder(
            [z1],
            input_is_latent=False,
            randomize_noise=False,
            return_latents=True,
            truncation=self.truncation,
            truncation_latent=self.net.latent_avg[0],
        )
        w1[:, 6:] = w2.detach()[:, 0]
        x1, w1_new = self.net.decoder(
            [w1],
            input_is_latent=True,
            randomize_noise=False,
            return_latents=True,
        )
        result = (
            ((x1.detach()[0].permute(1, 2, 0) + 1.0) * 127.5).cpu().numpy()[:, :, ::-1]
        )
        return (
            result,
            {
                "w1": w1_new.cpu().detach().numpy(),
                "w1_initial": w1_new.cpu().detach().numpy(),
            },
        )

    def reset(self, latents):
        w1, w1_initial, f1 = self.latents_to_tensor(latents)
        x1, w1_new, f1 = self.net.decoder(
            [w1_initial],
            input_is_latent=True,
            randomize_noise=False,
            return_feature_map=True,
            return_latents=True,
        )
        result = (
            ((x1.detach()[0].permute(1, 2, 0) + 1.0) * 127.5).cpu().numpy()[:, :, ::-1]
        )
        return (
            result,
            {
                "w1": w1_new.cpu().detach().numpy(),
                "w1_initial": w1_new.cpu().detach().numpy(),
            },
        )
