import gradio as gr

from .model_loader import Model
from PIL import Image
import cv2
import io

# models fron pretrained/latent_transformer folder
models_files = {
    "anime": "pretrained_models/latent_transformer/anime.pt",
    "car": "pretrained_models/latent_transformer/car.pt",
    "cat": "pretrained_models/latent_transformer/cat.pt",
    "church": "pretrained_models/latent_transformer/church.pt",
    "ffhq": "pretrained_models/latent_transformer/ffhq.pt",
}

models = {name: Model(path) for name, path in models_files.items()}


def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB))


def random_sample(model_name: str):
    model = models[model_name]
    img, latents = model.random_sample()
    pil_img = cv_to_pil(img)
    return pil_img, model_name, latents


def zoom(dx, dy, dz, model_state, latents_state):
    model = models[model_state]
    dx = dx
    dy = dy
    dz = dz
    sx = 100
    sy = 100
    stop_points = []
    img, latents_state = model.zoom(
        latents_state, dz, sxsy=[sx, sy], stop_points=stop_points
    )  # dz, sxsy=[sx, sy], stop_points=stop_points)
    pil_img = cv_to_pil(img)
    return pil_img, latents_state


def translate(dx, dy, dz, model_state, latents_state):
    model = models[model_state]

    dx = dx
    dy = dy
    dz = dz
    sx = 128
    sy = 128
    stop_points = []
    zi = False
    zo = False

    img, latents_state = model.translate(
        latents_state,
        [dx, dy],
        sxsy=[sx, sy],
        stop_points=stop_points,
        zoom_in=zi,
        zoom_out=zo,
    )

    pil_img = cv_to_pil(img)
    return pil_img, latents_state


def change_style(image: Image.Image, model_state, latents_state):
    model = models[model_state]
    img, latents_state = model.change_style(latents_state)
    pil_img = cv_to_pil(img)
    return pil_img, latents_state


def reset(model_state, latents_state):
    model = models[model_state]
    img, latents_state = model.reset(latents_state)
    pil_img = cv_to_pil(img)
    return pil_img, latents_state


with gr.Blocks() as block:
    model_state = gr.State(value="cat")
    latents_state = gr.State({})
    gr.Markdown("# UserControllableLT: User controllable latent transformer")
    gr.Markdown("## Select model")
    with gr.Row():
        with gr.Column():
            model_name = gr.Dropdown(
                choices=list(models_files.keys()),
                label="Select Pretrained Model",
                value="cat",
            )
            with gr.Row():
                button = gr.Button("Random sample")
                reset_btn = gr.Button("Reset")

            dx = gr.Slider(
                minimum=-128, maximum=128, step_size=0.1, label="dx", value=0.0
            )
            dy = gr.Slider(
                minimum=-128, maximum=128, step_size=0.1, label="dy", value=0.0
            )
            dz = gr.Slider(
                minimum=-128, maximum=128, step_size=0.1, label="dz", value=0.0
            )

            with gr.Row():
                change_style_bt = gr.Button("Change style")

        with gr.Column():
            image = gr.Image(type="pil", label="")
    button.click(
        random_sample, inputs=[model_name], outputs=[image, model_state, latents_state]
    )

    reset_btn.click(
        reset,
        inputs=[model_state, latents_state],
        outputs=[image, latents_state],
    )

    change_style_bt.click(
        change_style,
        inputs=[image, model_state, latents_state],
        outputs=[image, latents_state],
    )
    dx.change(
        translate,
        inputs=[dx, dy, dz, model_state, latents_state],
        outputs=[image, latents_state],
        show_progress=False,
    )
    dy.change(
        translate,
        inputs=[dx, dy, dz, model_state, latents_state],
        outputs=[image, latents_state],
        show_progress=False,
    )
    dz.change(
        zoom,
        inputs=[dx, dy, dz, model_state, latents_state],
        outputs=[image, latents_state],
        show_progress=False,
    )


block.launch()
