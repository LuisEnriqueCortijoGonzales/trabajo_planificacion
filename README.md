# Preprocesamiento del Dataset de Dinosaurios

Este documento explica **qué hace** el script `prepare_data.py`, **por qué** lo hace y **cómo** usarlo para generar los splits `train/val/test` listos para entrenamiento y evaluación.

> Este flujo cumple con los requisitos del Trabajo 6 (entrenar un modelo, guardar artefactos y preparar una API local), dejando la data en un formato reproducible y limpio para el entrenamiento y demo.

---

## Objetivo del preprocesamiento

1. **Estandarizar** la estructura del dataset a un formato clásico de visión por computadora:
   ```
   datasets/dinosaurs/
     ├─ train/<Clase>/*.jpg
     ├─ val/<Clase>/*.jpg
     └─ test/<Clase>/*.jpg
   ```

2. **Asegurar calidad**: descartar imágenes corruptas o demasiado pequeñas y eliminar duplicados exactos.

3. **Crear splits reproducibles** (train/val/test) con proporciones configurables.

4. **Dejar metadatos** útiles para el entrenamiento y la API: clases, índices, estadísticas.

---

## ¿Qué hace exactamente `prepare_data.py`?

### 1) Descubrimiento de clases por carpetas
- **Qué**: Detecta cada subcarpeta dentro de `--src_dir` como una **clase** (p. ej. `Ankylosaurus`, `Tyrannosaurus`).
- **Por qué**: Es el estándar en `torchvision.datasets.ImageFolder` y evita hardcodear nombres de clases.

### 2) Validación de imágenes (integridad y tamaño mínimo)
- **Qué**: Abre cada archivo con **PIL**:
  - Llama a `im.verify()` para comprobar integridad/encabezados.
  - Reabre para leer `width × height` y descarta imágenes con lados < `--min_size` (default: 64 px).
- **Por qué**:
  - Evitamos fallos en entrenamiento por **archivos corruptos**.
  - Imágenes **demasiado pequeñas** suelen ser miniaturas/íconos sin detalle → ruido para el modelo.

### 3) Deduplicado por hash SHA-1 (opcional)
- **Qué**: Calcula SHA-1 de los bytes de cada imagen y **omite** imágenes cuyo hash ya apareció.
- **Por qué**:
  - El **duplicado exacto** sesga el modelo (data leakage entre splits o sobreajuste).
  - Mantener clases balanceadas mejora la generalización.
- **Notas**: Se puede desactivar con `--no_dedup`.

### 4) Split estratificado en **train/val/test**
- **Qué**: Para **cada clase**:
  - Mezcla aleatoriamente con semilla (`--seed`).
  - Divide según `--train_ratio`, `--val_ratio`, `--test_ratio` (por defecto **0.70/0.15/0.15**).
- **Por qué**:
  - Asegura que **todas las clases** aparezcan en los tres splits.
  - Mantiene **distribución por clase** similar entre splits.
  - Permite un **test final** “intocable” para medir desempeño real.

### 5) Copia o symlink de archivos
- **Qué**: Escribe los archivos en `datasets/dinosaurs/{train,val,test}/<Clase>/...` usando:
  - `--mode copy` (por defecto): copia física de archivos.
  - `--mode link`: intenta **symlink** (ahorra espacio).
- **Por qué**:
  - **copy** es robusto y funciona en todos los sistemas (Windows, macOS, Linux).
  - **link** es útil cuando el dataset ocupa mucho espacio (pero en Windows puede requerir permisos de admin).

### 6) Metadatos generados
- **`class_index.json`**: mapeo `índice → nombre de clase` (p. ej. `0: "Ankylosaurus"`).  
- **`classes.txt`**: lista de clases (una por línea).
- **`dataset_stats.json`**: resumen de conteos por clase y totales.

---

## Parámetros importantes

- `--src_dir` *(obligatorio)*: carpeta origen con subcarpetas por clase.  
- `--out_dir` *(default: `datasets/dinosaurs`)*: carpeta destino para los splits.
- `--train_ratio --val_ratio --test_ratio` *(default: 0.70/0.15/0.15)*: deben **sumar 1.0**.
- `--min_size` *(default: 64)*: descarta imágenes con cualquiera de los lados < `min_size`.
- `--no_dedup`: desactiva deduplicado por SHA-1.
- `--mode copy|link` *(default: copy)*: método de escritura de archivos en la salida.
- `--max_per_class` *(default: 0 = sin límite)*: útil para prototipado/entrenamientos rápidos (cap por clase).
- `--seed` *(default: 42)*: asegura reproducibilidad del split.

---

## Uso

### PowerShell (Windows)
```powershell
python prepare_data.py --src_dir dinosaur_dataset --out_dir datasets/dinosaurs --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --mode copy
```

### Bash (Linux/macOS)
```bash
python prepare_data.py --src_dir dinosaur_dataset --out_dir datasets/dinosaurs   --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --mode copy
```

**Salida esperada**:
```
datasets/dinosaurs/
  ├─ train/<Clase>/*.jpg
  ├─ val/<Clase>/*.jpg
  └─ test/<Clase>/*.jpg
  ├─ class_index.json
  ├─ classes.txt
  └─ dataset_stats.json
```

---

## Decisiones de diseño

- **Validación + tamaño mínimo**: filtrar errores antes de entrenar evita crashes y basura visual.
- **Deduplicado global**: reduce fuga de información y sobreajuste por repetición.
- **Split por clase** (estratificado): asegura representación de todas las clases en train/val/test.
- **Ratios 70/15/15**: prácticos para prototipos rápidos.
- **Semilla fija**: reproducibilidad total.
- **copy vs link**: `copy` es más compatible; `link` ideal si necesitas ahorrar espacio.

---

## Errores comunes

- `unrecognized arguments: \` en PowerShell: usar una sola línea en Windows.
- Symlink fallido en Windows: mantener `--mode copy` o correr PowerShell como admin.
- Ratios no suman 1.0: el script lanza error.
- Clases vacías: carpetas con solo imágenes corruptas/miniaturas se omiten (aviso en consola).

---

## Próximo paso

Con los splits creados, tu dataset está **listo para entrenamiento**.  
El siguiente paso es implementar `train.py` con **ResNet18 preentrenada**, congelar el backbone y guardar `artifacts/model.pth` + `artifacts/class_index.json`.
