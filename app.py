import gradio as gr
import torch
from torchvision.transforms import v2

from src.model import load_model
from src.data import eval_tf

model, ckpt = load_model(device="cpu")
classes = ckpt["classes"]

@torch.no_grad()
def predict(img):
    x = eval_tf(v2.functional.pil_to_tensor(img.convert("RGB"))).unsqueeze(0)
    probs = model(x).softmax(1)[0]
    return {c: float(p) for c, p in zip(classes, probs)}

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Ta bilde av en blomst"),
    outputs=gr.Label(num_top_classes=3),
    title="BloMLster",
).launch(share=True)