import argparse, json, os, time, random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

# -----------------------
# Utils
# -----------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_transforms(img_size=256, strong_aug=True):
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    train_tf = [
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if strong_aug:
        train_tf += [
            transforms.RandomApply([transforms.ColorJitter(0.3,0.3,0.3,0.1)], p=0.8),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
            transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.3),
        ]
    train_tf += [transforms.ToTensor(), transforms.Normalize(mean,std)]
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    return transforms.Compose(train_tf), eval_tf

def choose_model(name, num_classes):
    name = name.lower()
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features; m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_f = m.fc.in_features; m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Modelo no soportado: {name}")
    return m

def get_class_weights(targets, num_classes):
    cnt = Counter(targets)
    total = sum(cnt.values())
    # inverso de la frecuencia
    weights = torch.tensor([total / (num_classes * max(1, cnt.get(i,0))) for i in range(num_classes)], dtype=torch.float32)
    return weights

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, (y_a, y_b), lam

def cutmix_data(x, y, alpha=0.2):
    if alpha <= 0: return x, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.size()
    index = torch.randperm(B, device=x.device)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
    cx = np.random.randint(W); cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W); y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W); y2 = np.clip(cy + cut_h // 2, 0, H)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, (y_a, y_b), lam

def criterion_with_ls(outputs, targets, label_smoothing=0.0, class_weights=None):
    if isinstance(targets, tuple):  # mixup/cutmix
        y_a, y_b, lam = targets
        return lam * nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)(outputs, y_a) + \
               (1 - lam) * nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)(outputs, y_b)
    else:
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)(outputs, targets)

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0; correct = 0; total = 0
    ce = nn.CrossEntropyLoss()
    for x,y in loader:
        x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum/max(1,total), correct/max(1,total)

@torch.inference_mode()
def evaluate_full(model, loader, device, num_classes, save_path_png):
    model.eval()
    ce = nn.CrossEntropyLoss()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    loss_sum=0; total=0; correct=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += loss.item()*y.size(0); total += y.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        for t,p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[t,p]+=1
    acc = correct/max(1,total)
    # plot CM
    plt.figure(figsize=(8,7))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'Confusion Matrix (acc={acc:.3f})')
    plt.colorbar()
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig(save_path_png, dpi=150)
    plt.close()
    return loss_sum/max(1,total), acc, cm.tolist()

# -----------------------
# Train Loop (2 fases)
# -----------------------
def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | img={args.img_size} | AMP={args.amp}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.img_size, strong_aug=not args.light_aug)
    train_ds = datasets.ImageFolder(data_dir/"train", transform=train_tf)
    val_ds   = datasets.ImageFolder(data_dir/"val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(data_dir/"test",  transform=eval_tf)

    num_classes = len(train_ds.classes)
    print(f"Clases ({num_classes}): {train_ds.classes}")
    idx2cls = {i:c for i,c in enumerate(train_ds.classes)}
    (out_dir/"class_index.json").write_text(json.dumps(idx2cls, indent=2), encoding="utf-8")

    # Sampler con pesos (si hay desbalance)
    targets = [y for _,y in datasets.ImageFolder(data_dir/"train").imgs]  # sin tf para leer targets
    class_counts = Counter(targets)
    print("Train class counts:", dict(class_counts))
    if args.weighted_sampler:
        weights_per_class = 1.0 / np.maximum(1, np.array([class_counts.get(i,0) for i in range(num_classes)]))
        sample_weights = [weights_per_class[y] for y in targets]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None; shuffle = True

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # Modelo
    model = choose_model(args.model, num_classes).to(device)

    # ----- Fase 1: entrenar SOLO la cabeza -----
    for p in model.parameters():
        p.requires_grad = False
    for p in model.parameters():
        if p.dim() == 2 and p.size(0) == num_classes:  # intento general, pero…
            break
    for p in model.modules():
        pass
    # habilitamos la cabeza explícitamente
    if hasattr(model, "fc"):
        for p in model.fc.parameters(): p.requires_grad = True
    elif hasattr(model, "classifier"):
        for p in model.classifier.parameters(): p.requires_grad = True

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr_head, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_ld))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_head, epochs=args.epochs_head, steps_per_epoch=steps_per_epoch
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    ce_weights = None
    if args.class_weights:
        ce_weights = get_class_weights(targets, num_classes).to(device)

    best_val = 0.0; patience = args.es_patience; es_counter = 0
    history = []

    def train_one_epoch(epoch, mixup_alpha, cutmix_alpha, label_smooth):
        model.train(); t0=time.time(); loss_sum=0; seen=0
        for x,y in train_ld:
            x,y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            # Mixup/Cutmix (prioridad cutmix si ambos > 0)
            if cutmix_alpha>0:
                x, (ya,yb), lam = cutmix_data(x,y,cutmix_alpha)
                targets_mix = (ya,yb,lam)
            elif mixup_alpha>0:
                x, (ya,yb), lam = mixup_data(x,y,mixup_alpha)
                targets_mix = (ya,yb,lam)
            else:
                targets_mix = y

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(x)
                loss = criterion_with_ls(logits, targets_mix, label_smoothing=label_smooth, class_weights=ce_weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            scheduler.step()

            bs = y.size(0)
            loss_sum += loss.item()*bs; seen += bs
        return loss_sum/max(1,seen), time.time()-t0

    print(f"\n=== Fase 1: cabeza ({args.epochs_head} epochs, lr={args.lr_head}) ===")
    for epoch in range(1, args.epochs_head+1):
        tr_loss, dt = train_one_epoch(epoch, args.mixup_alpha, args.cutmix_alpha, args.label_smoothing)
        val_loss, val_acc = evaluate(model, val_ld, device)
        print(f"[Head] Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")
        history.append({"phase":"head","epoch":epoch,"train_loss":tr_loss,"val_loss":val_loss,"val_acc":val_acc})
        if val_acc > best_val:
            best_val = val_acc; es_counter = 0
            torch.save({"model_state": model.state_dict(), "classes": train_ds.classes, "img_size": args.img_size}, out_dir/"model_head.pth")
        else:
            es_counter += 1
            if es_counter >= patience:
                print("EarlyStopping (fase 1)"); break

    # ----- Fase 2: fine-tuning completo -----
    for p in model.parameters(): p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_ft, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr_ft, epochs=args.epochs_ft, steps_per_epoch=steps_per_epoch
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    es_counter = 0

    print(f"\n=== Fase 2: fine-tuning ({args.epochs_ft} epochs, lr={args.lr_ft}) ===")
    best_val = 0.0
    for epoch in range(1, args.epochs_ft+1):
        tr_loss, dt = train_one_epoch(epoch, args.mixup_alpha, args.cutmix_alpha, args.label_smoothing)
        val_loss, val_acc = evaluate(model, val_ld, device)
        print(f"[FT ] Epoch {epoch:02d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | {dt:.1f}s")
        history.append({"phase":"ft","epoch":epoch,"train_loss":tr_loss,"val_loss":val_loss,"val_acc":val_acc})
        if val_acc > best_val:
            best_val = val_acc; es_counter = 0
            torch.save({"model_state": model.state_dict(), "classes": train_ds.classes, "img_size": args.img_size}, out_dir/"model.pth")
        else:
            es_counter += 1
            if es_counter >= patience:
                print("EarlyStopping (fase 2)"); break

    # cargar mejor y evaluar en test + CM
    ckpt = torch.load(out_dir/"model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc, cm = evaluate_full(model, test_ld, device, num_classes, save_path_png=str(out_dir/"confusion_matrix.png"))
    metrics = {"best_val_acc": float(best_val), "test_loss": float(test_loss), "test_acc": float(test_acc)}
    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir/"history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\n✅ Done. best_val_acc={best_val:.4f} | test_acc={test_acc:.4f}")
    print(f"Métricas: {out_dir/'metrics.json'} | CM: {out_dir/'confusion_matrix.png'} | Pesos: {out_dir/'model.pth'}")

# -----------------------
# Main
# -----------------------
def parse_args():
    ap = argparse.ArgumentParser("DinoClassifier Advanced Trainer")
    ap.add_argument("--data_dir", default="datasets/dinosaurs")
    ap.add_argument("--out_dir", default="artifacts")

    ap.add_argument("--model", default="resnet50",
                    choices=["resnet18","resnet50","mobilenet_v3_large","efficientnet_b0"])
    ap.add_argument("--img_size", type=int, default=256)

    ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_ft", type=int, default=20)

    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--lr_ft", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=0.0)  # usa uno u otro; CutMix suele ir mejor con imgs grandes
    ap.add_argument("--label_smoothing", type=float, default=0.1)

    ap.add_argument("--class_weights", action="store_true", help="Pondera pérdida por clase (CrossEntropy)")
    ap.add_argument("--weighted_sampler", action="store_true", help="Sampler ponderado para desbalance")

    ap.add_argument("--light_aug", action="store_true", help="Usa augmentations suaves")
    ap.add_argument("--es_patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)
