import os, json
from pathlib import Path
from typing import List
import gradio as gr
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "artifacts/model.pth")
CLASS_INDEX = os.getenv("CLASS_INDEX", "artifacts/class_index.json")
DEFAULT_IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_eval_tf(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def infer_arch_from_state_dict(sd_keys: List[str]) -> str:
    ks = set(sd_keys)
    if any(k.startswith("fc.") for k in ks): return "resnet_auto"
    if any(k.startswith("classifier.3") for k in ks): return "mobilenet_v3_large"
    if any(k.startswith("classifier.1") for k in ks): return "efficientnet_b0"
    return "resnet18"

def build_model_for_arch(arch: str, num_classes: int):
    arch = arch.lower()
    if arch == "resnet50":
        m = models.resnet50(weights=None); in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes)); return m, "resnet"
    if arch in ("resnet18","resnet_auto"):
        m = models.resnet18(weights=None); in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes)); return m, "resnet"
    if arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None); in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes); return m, "mobilenetv3"
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None); in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes); return m, "efficientnet"
    m = models.resnet18(weights=None); in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes)); return m, "resnet"

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
    if classes is None: raise RuntimeError("No classes")
    img_size = int(ckpt.get("img_size", DEFAULT_IMG_SIZE))
    sd = ckpt["model_state"]
    arch_hint = infer_arch_from_state_dict(list(sd.keys()))
    for arch in (["resnet50","resnet18"] if arch_hint=="resnet_auto" else [arch_hint]):
        model, family = build_model_for_arch(arch, num_classes=len(classes))
        try:
            model.load_state_dict(sd, strict=True); model.to(device).eval()
            return model, family, classes, img_size
        except Exception: pass
    model, family = build_model_for_arch("resnet18", num_classes=len(classes))
    model.load_state_dict(sd, strict=False); model.to(device).eval()
    return model, family, classes, img_size

MODEL, FAMILY, CLASSES, IMG_SIZE = load_model(WEIGHTS_PATH, CLASS_INDEX)
EVAL_TF = build_eval_tf(IMG_SIZE)
IDX2CLS = {i: c for i, c in enumerate(CLASSES)}

@torch.inference_mode()
def predict_topk(pil_img: Image.Image, top_k: int = 5):
    if pil_img is None: return []
    x = EVAL_TF(pil_img).unsqueeze(0).to(device)
    probs = torch.softmax(MODEL(x), dim=1)[0]
    top_k = int(min(max(int(top_k), 1), len(IDX2CLS)))
    topk = torch.topk(probs, k=top_k)
    pairs = [(IDX2CLS[i.item()], float(probs[i])) for i in topk.indices]
    return [[c, f"{p*100:.2f}%"] for c, p in pairs]

def health():
    return {"status":"ok","device":str(device)}

def labels():
    return {"classes": CLASSES, "num_classes": len(CLASSES)}

def metadata():
    return {"model_family": FAMILY, "img_size": IMG_SIZE, "weights_path": WEIGHTS_PATH}

with gr.Blocks(title="DinoClassifier — Top-K") as demo:
    gr.Markdown(f"# 🦖 DinoClassifier — Top-K\n**Modelo**: {FAMILY}  |  **Clases**: {len(CLASSES)}  |  **img_size**: {IMG_SIZE}")
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="pil", label="Imagen", height=360)
                    topk = gr.Slider(minimum=1, maximum=len(CLASSES), value=5, step=1, label="Top-K")
                    btn = gr.Button("Predecir", variant="primary")
                with gr.Column(scale=1):
                    table = gr.Dataframe(headers=["Clase","Probabilidad"], label="Top-K", interactive=False)
            btn.click(fn=predict_topk, inputs=[img, topk], outputs=table, api_name="predict")
        with gr.Tab("Info API"):
            with gr.Row():
                b1 = gr.Button("Health"); out1 = gr.JSON()
            with gr.Row():
                b2 = gr.Button("Labels"); out2 = gr.JSON()
            with gr.Row():
                b3 = gr.Button("Metadata"); out3 = gr.JSON()
            b1.click(fn=health, inputs=None, outputs=out1, api_name="health")
            b2.click(fn=labels, inputs=None, outputs=out2, api_name="labels")
            b3.click(fn=metadata, inputs=None, outputs=out3, api_name="metadata")
    img_api = gr.Image(type="pil", visible=False)
    k_api = gr.Number(value=5, visible=False, precision=0)
    btn_api = gr.Button(visible=False)
    btn_api.click(fn=predict_topk, inputs=[img_api, k_api], outputs=gr.JSON(), api_name="predict_json")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)

