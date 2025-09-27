import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "artifacts/model.pth")
CLASS_INDEX = os.getenv("CLASS_INDEX", "artifacts/class_index.json")
DEFAULT_IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_eval_tf(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

class GradCAM:
    def __init__(self, model, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        score = logits[:, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward(retain_graph=True)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze().cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam, class_idx

def overlay_heatmap(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.35):
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = (alpha * heatmap + (1 - alpha) * rgb[..., ::-1]).astype(np.uint8) 
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def infer_arch_from_state_dict(sd_keys: List[str]) -> str:
    keys = set(sd_keys)
    if any(k.startswith("fc.") for k in keys):
        return "resnet_auto"
    if any(k.startswith("classifier.3") for k in keys):
        return "mobilenet_v3_large"
    if any(k.startswith("classifier.1") for k in keys):
        return "efficientnet_b0"
    return "resnet18"

def build_model_for_arch(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
        return m, "resnet"
    if arch == "resnet18" or arch == "resnet_auto":
        m = models.resnet18(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
        return m, "resnet"
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m, "mobilenetv3"
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m, "efficientnet"
    m = models.resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    return m, "resnet"

def pick_target_conv_layer(model, family: str) -> nn.Module:
    if family == "resnet":
        return model.layer4[-1].conv2
    if family == "mobilenetv3":
        conv = None
        for m in model.features.modules():
            if isinstance(m, nn.Conv2d):
                conv = m
        return conv
    if family == "efficientnet":
        conv = None
        for m in model.features.modules():
            if isinstance(m, nn.Conv2d):
                conv = m
        return conv
    conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv = m
    return conv

def load_model(weights_path: str, class_index_path: str = None):
    ckpt = torch.load(weights_path, map_location=device)
    classes = ckpt.get("classes")
    if classes is None and class_index_path and Path(class_index_path).exists():
        with open(class_index_path, "r", encoding="utf-8") as f:
            idx2cls = json.load(f)
        if isinstance(list(idx2cls.keys())[0], str):
            classes = [idx2cls[str(i)] for i in range(len(idx2cls))]
        else:
            classes = [idx2cls[i] for i in range(len(idx2cls))]
    if classes is None:
        raise RuntimeError("No se encontraron 'classes' en el checkpoint ni en class_index.json")

    img_size = int(ckpt.get("img_size", DEFAULT_IMG_SIZE))
    sd = ckpt["model_state"]
    arch_hint = infer_arch_from_state_dict(list(sd.keys()))

    tried = []
    for arch in ([ "resnet50", "resnet18" ] if arch_hint == "resnet_auto" else [arch_hint]):
        model, family = build_model_for_arch(arch, num_classes=len(classes))
        try:
            model.load_state_dict(sd, strict=True)
            print(f"[frontend] Modelo cargado como {arch}.")
            model.to(device).eval()
            return model, family, classes, img_size
        except Exception as e:
            tried.append((arch, str(e)))
            continue

    model, family = build_model_for_arch("resnet18", num_classes=len(classes))
    model.load_state_dict(sd, strict=False) 
    print("[frontend] Atención: carga 'strict=False' como resnet18. Intentos previos:", tried)
    model.to(device).eval()
    return model, family, classes, img_size

MODEL, FAMILY, CLASSES, IMG_SIZE = load_model(WEIGHTS_PATH, CLASS_INDEX)
EVAL_TF = build_eval_tf(IMG_SIZE)
IDX2CLS = {i: c for i, c in enumerate(CLASSES)}
TARGET_LAYER = pick_target_conv_layer(MODEL, FAMILY)
GCAM = GradCAM(MODEL, TARGET_LAYER)

@torch.inference_mode()
def predict_image(pil_img: Image.Image, top_k: int = 5, explain: bool = False):
    if pil_img is None:
        return "Sin imagen", [], None
    x = EVAL_TF(pil_img).unsqueeze(0).to(device)
    logits = MODEL(x)
    probs = torch.softmax(logits, dim=1)[0]
    top_k = int(min(max(top_k, 1), len(IDX2CLS)))
    topk = torch.topk(probs, k=top_k)
    pairs = [(IDX2CLS[i.item()], float(probs[i])) for i in topk.indices]
    label = f"{pairs[0][0]} ({pairs[0][1]*100:.2f}%)"

    heatmap_img = None
    if explain:
        # Grad-CAM necesita gradientes
        with torch.enable_grad():
            x.requires_grad_(True)
            cam, class_idx = GCAM(x, class_idx=topk.indices[0].item())
        rgb = np.array(pil_img.convert("RGB"))
        heatmap_img = overlay_heatmap(rgb, cam)

    table = [[c, f"{p*100:.2f}%"] for c, p in pairs]
    return label, table, heatmap_img

with gr.Blocks(title="DinoClassifier") as demo:
    gr.Markdown("# 🦖 DinoClassifier — Gradio Demo")
    gr.Markdown(
        f"Modelo: **{FAMILY}** | Clases: **{len(CLASSES)}** | img_size: **{IMG_SIZE}**"
    )
    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(type="pil", label="Imagen", height=360)
            topk = gr.Slider(minimum=1, maximum=len(CLASSES), value=5, step=1, label="Top-K")
            explain = gr.Checkbox(value=False, label="Explicar (Grad-CAM)")
            btn = gr.Button("Predecir", variant="primary")
        with gr.Column(scale=1):
            top1 = gr.Label(label="Predicción (Top-1)")
            table = gr.Dataframe(headers=["Clase", "Probabilidad"], label="Top-K", interactive=False)
            heat = gr.Image(label="Grad-CAM (si está activado)")

    btn.click(fn=predict_image, inputs=[img, topk, explain], outputs=[top1, table, heat], api_name="predict")

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, share=False)
