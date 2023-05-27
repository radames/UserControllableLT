import gradio as gr
import sys

sys.path.append(".")
sys.path.append("..")
from model_loader import Model
from inversion import InversionModel
from PIL import Image
import cv2
from huggingface_hub import snapshot_download
import json

# disable if running on another environment
RESIZE = True

models_path = snapshot_download(repo_id="radames/UserControllableLT", repo_type="model")


# models fron pretrained/latent_transformer folder
models_files = {
    "anime": "anime.pt",
    "car": "car.pt",
    "cat": "cat.pt",
    "church": "church.pt",
    "ffhq": "ffhq.pt",
}

models = {name: Model(models_path + "/" + path) for name, path in models_files.items()}
inversion_model = InversionModel(models_path + "/psp_ffhq_encode.pt", models_path + "/shape_predictor_68_face_landmarks.dat")

canvas_html = """<draggan-canvas id="canvas-root" style='display:flex;max-width: 500px;margin: 0 auto;'></draggan-canvas>"""
load_js = """
async () => {
  const script = document.createElement('script');
  script.type = "module"
  script.src = "file=custom_component.js"
  document.head.appendChild(script);
}
"""
image_change = """
async (base64img) => {
  const canvasEl = document.getElementById("canvas-root");
  canvasEl.loadBase64Image(base64img);
}   
"""
reset_stop_points = """
async () => {
  const canvasEl = document.getElementById("canvas-root");
  canvasEl.resetStopPoints();
}
"""

default_dxdysxsy = json.dumps(
    {"dx": 1, "dy": 0, "sx": 128, "sy": 128, "stopPoints": []}
)


def cv_to_pil(img):
    img = Image.fromarray(cv2.cvtColor(img.astype("uint8"), cv2.COLOR_BGR2RGB))
    if RESIZE:
        img = img.resize((128, 128))
    return img


def random_sample(model_name: str):
    model = models[model_name]
    img, latents = model.random_sample()
    img_pil = cv_to_pil(img)
    return img_pil, model_name, latents


def transform(model_state, latents_state, dxdysxsy=default_dxdysxsy, dz=0):
    if "w1" not in latents_state or "w1_initial" not in latents_state:
        raise gr.Error("Generate a random sample first")

    data = json.loads(dxdysxsy)

    model = models[model_state]
    dx = int(data["dx"])
    dy = int(data["dy"])
    sx = int(data["sx"])
    sy = int(data["sy"])
    stop_points = [[int(x), int(y)] for x, y in data["stopPoints"]]
    img, latents_state = model.transform(
        latents_state, dz, dxy=[dx, dy], sxsy=[sx, sy], stop_points=stop_points
    )
    img_pil = cv_to_pil(img)
    return img_pil, latents_state


def change_style(image: Image.Image, model_state, latents_state):
    model = models[model_state]
    img, latents_state = model.change_style(latents_state)
    img_pil = cv_to_pil(img)
    return img_pil, latents_state


def reset(model_state, latents_state):
    model = models[model_state]
    img, latents_state = model.reset(latents_state)
    img_pil = cv_to_pil(img)
    return img_pil, latents_state


def image_click(evt: gr.SelectData):
    click_pos = evt.index
    return click_pos


with gr.Blocks() as block:
    model_state = gr.State(value="cat")
    latents_state = gr.State({})
    gr.Markdown(
        """# UserControllableLT: User Controllable Latent Transformer
Unofficial Gradio Demo

**Author**: Yuki Endo\\
**Paper**:  [2208.12408](http://arxiv.org/abs/2208.12408)\\
**Code**: [UserControllableLT](https://github.com/endo-yuki-t/UserControllableLT)

<small>
Double click to add or remove stop points.
<small>
"""
    )

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
                change_style_bt = gr.Button("Change style")
            dxdysxsy = gr.Textbox(
                label="dxdysxsy",
                value=default_dxdysxsy,
                elem_id="dxdysxsy",
                visible=False,
            )
            dz = gr.Slider(
                minimum=-15, maximum=15, step_size=0.01, label="zoom", value=0.0
            )
            image = gr.Image(type="pil", visible=False, preprocess=False)

        with gr.Column():
            html = gr.HTML(canvas_html, label="output")

    button.click(
        random_sample, inputs=[model_name], outputs=[image, model_state, latents_state]
    )
    reset_btn.click(
        reset,
        inputs=[model_state, latents_state],
        outputs=[image, latents_state],
        queue=False,
    ).then(None, None, None, _js=reset_stop_points, queue=False)

    change_style_bt.click(
        change_style,
        inputs=[image, model_state, latents_state],
        outputs=[image, latents_state],
    )
    dxdysxsy.change(
        transform,
        inputs=[model_state, latents_state, dxdysxsy, dz],
        outputs=[image, latents_state],
        show_progress=False,
    )
    dz.change(
        transform,
        inputs=[model_state, latents_state, dxdysxsy, dz],
        outputs=[image, latents_state],
        show_progress=False,
    )
    image.change(None, inputs=[image], outputs=None, _js=image_change)
    block.load(None, None, None, _js=load_js)
    block.load(
        random_sample, inputs=[model_name], outputs=[image, model_state, latents_state]
    )

block.queue(api_open=False)
block.launch(show_api=False)
