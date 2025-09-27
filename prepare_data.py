import argparse
import os
import sys
import json
import shutil
import random
import hashlib
from pathlib import Path
from typing import List, Tuple
from PIL import Image

# Extensiones de imagen soportadas
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def verify_image(p: Path, min_size: int = 64) -> bool:
    """
    True si la imagen se puede abrir y tiene ambos lados >= min_size.
    """
    try:
        with Image.open(p) as im:
            im.verify()  # chequeo rápido
        # Reabrimos para leer tamaño (verify() cierra el archivo)
        with Image.open(p) as im2:
            w, h = im2.size
        return (w >= min_size and h >= min_size)
    except Exception:
        return False

def file_sha1(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def copy_or_link(src: Path, dst: Path, mode: str = "copy"):
    """
    Copia o crea symlink (si es posible). En Windows, symlink puede fallar sin permisos de admin.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            if not dst.exists():
                os.symlink(src.resolve(), dst)
        except (NotImplementedError, OSError):
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)

def stratified_split_3(items: List[Path], ratios: Tuple[float, float, float], seed: int):
    """
    Divide en train/val/test según las proporciones dadas.
    Asegura que las tres listas cubran todos los elementos (lo que sobra va a test).
    """
    random.Random(seed).shuffle(items)
    n = len(items)
    if n == 0:
        return [], [], []
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    # Lo que reste va para test:
    n_test  = n - n_train - n_val
    # Correcciones para clases muy pequeñas:
    if n_train == 0 and n >= 1:
        n_train = 1
        if n_val > 0:
            n_val = max(0, n_val - 1)
        else:
            n_test = max(0, n_test - 1)
    train = items[:n_train]
    val   = items[n_train:n_train+n_val]
    test  = items[n_train+n_val:]
    return train, val, test

def prepare_dataset(
    src_dir: Path,
    out_dir: Path,
    ratios=(0.7, 0.15, 0.15),
    seed: int = 42,
    min_size: int = 64,
    dedup: bool = True,
    mode: str = "copy",
    max_per_class: int = 0
):
    assert src_dir.exists(), f"Fuente no encontrada: {src_dir}"

    # Crear carpetas de salida
    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    # Detectar clases por subcarpetas
    class_names = sorted([d.name for d in src_dir.iterdir() if d.is_dir()])
    if not class_names:
        print("No se encontraron carpetas de clase en la fuente.", file=sys.stderr)
        sys.exit(1)

    # Estadísticas
    stats = {
        "source": str(src_dir),
        "output": str(out_dir),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "seed": seed,
        "min_size": min_size,
        "mode": mode,
        "dedup": dedup,
        "classes": {},
        "total": {"raw": 0, "valid": 0, "duplicates": 0, "too_small_or_broken": 0}
    }

    global_hashes = set()

    for cls in class_names:
        cls_dir = src_dir / cls
        raw_imgs = [p for p in cls_dir.iterdir() if is_image_file(p)]
        stats["total"]["raw"] += len(raw_imgs)

        valid_imgs = []
        dup_count = 0
        bad_count = 0

        # Validación + deduplicado global
        for p in raw_imgs:
            if not verify_image(p, min_size=min_size):
                bad_count += 1
                continue
            if dedup:
                h = file_sha1(p)
                if h in global_hashes:
                    dup_count += 1
                    continue
                global_hashes.add(h)
            valid_imgs.append(p)

        # Límite por clase (para prototipos rápidos)
        if max_per_class and len(valid_imgs) > max_per_class:
            valid_imgs = valid_imgs[:max_per_class]

        stats["classes"][cls] = {
            "raw": len(raw_imgs),
            "valid_after_checks": len(valid_imgs),
            "duplicates": dup_count,
            "too_small_or_broken": bad_count
        }
        stats["total"]["valid"] += len(valid_imgs)
        stats["total"]["duplicates"] += dup_count
        stats["total"]["too_small_or_broken"] += bad_count

        if len(valid_imgs) == 0:
            print(f"[WARN] Clase '{cls}' sin imágenes válidas; se omite del split.", file=sys.stderr)
            continue

        # Split 3 vías por clase
        train_imgs, val_imgs, test_imgs = stratified_split_3(valid_imgs, ratios, seed)

        # Copiar/enlazar
        for split, subset in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
            for p in subset:
                dst = out_dir / split / cls / p.name
                copy_or_link(p, dst, mode=mode)

    # Guardar mapping de clases y stats
    idx_map = {i: c for i, c in enumerate(class_names)}
    with open(out_dir / "class_index.json", "w", encoding="utf-8") as f:
        json.dump(idx_map, f, indent=2, ensure_ascii=False)
    with open(out_dir / "classes.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    with open(out_dir / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Resumen bonito en consola
    print("\n=== RESUMEN DATASET ===")
    print(f"Clases detectadas: {len(class_names)}")
    for cls in class_names:
        c = stats["classes"].get(cls, {})
        if not c:
            print(f" - {cls}: (omitida, 0 válidas)")
            continue
        print(f" - {cls}: raw={c['raw']} | válidas={c['valid_after_checks']} | dup={c['duplicates']} | rotas/pequeñas={c['too_small_or_broken']}")
    t = stats["total"]
    print("\nTotales:",
          f"raw={t['raw']},",
          f"válidas={t['valid']},",
          f"dup={t['duplicates']},",
          f"rotas/pequeñas={t['too_small_or_broken']}")
    print(f"\nSplit creado en: {out_dir}")
    print(" - train/: imágenes por clase")
    print(" - val/:   imágenes por clase")
    print(" - test/:  imágenes por clase")
    print(" - class_index.json, classes.txt, dataset_stats.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preparar dataset (validación + deduplicado + split train/val/test).")
    ap.add_argument("--src_dir", type=str, required=True,
                    help="Ruta a la carpeta origen (p.ej. dinosaur_dataset).")
    ap.add_argument("--out_dir", type=str, default="datasets/dinosaurs",
                    help="Carpeta de salida para el split.")
    ap.add_argument("--train_ratio", type=float, default=0.7,
                    help="Proporción para entrenamiento.")
    ap.add_argument("--val_ratio", type=float, default=0.15,
                    help="Proporción para validación.")
    ap.add_argument("--test_ratio", type=float, default=0.15,
                    help="Proporción para test.")
    ap.add_argument("--seed", type=int, default=42, help="Semilla para el split.")
    ap.add_argument("--min_size", type=int, default=64, help="Mínimo ancho/alto permitido.")
    ap.add_argument("--no_dedup", action="store_true", help="Desactiva eliminación de duplicados por hash.")
    ap.add_argument("--mode", choices=["copy", "link"], default="copy",
                    help="copy = copiar archivos (seguro), link = symlink (ahorra espacio; puede fallar en Windows).")
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="Máximo de imágenes por clase (0 = sin límite). Útil para prototipado rápido.")
    args = ap.parse_args()

    # Validación de ratios
    s = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones deben sumar 1.0, suma={s}")

    prepare_dataset(
        src_dir=Path(args.src_dir),
        out_dir=Path(args.out_dir),
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
        min_size=args.min_size,
        dedup=not args.no_dedup,
        mode=args.mode,
        max_per_class=args.max_per_class
    )
