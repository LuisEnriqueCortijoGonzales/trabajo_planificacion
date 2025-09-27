# 🦖 DinoClassifier — Clasificación de Dinosaurios con Deep Learning

Este proyecto implementa un sistema de clasificación automática de dinosaurios a partir de imágenes, utilizando técnicas avanzadas de deep learning y un frontend interactivo para facilitar su uso.
link del dataset: https://www.kaggle.com/datasets/larserikrisholm/dinosaur-image-dataset-15-species/suggestions

## Descripción General

Este proyecto implementa un sistema de clasificación automática de dinosaurios a partir de imágenes, combinando **preprocesamiento de datos**, **entrenamiento de modelos CNN** y un **frontend interactivo en Gradio**.

El flujo principal consiste en:

1. **Preparar los datos** (validación, limpieza, quitar duplicados y separación para train, val y test).
2. **Entrenar modelos de IA**.
3. **Desplegar un frontend interactivo**.

---


## Integrantes del Equipo

Un equipo diverso y apasionado de estudiantes está detrás de este proyecto, listo para sumergirse en el reino de las búsquedas y de la indexación multidimensional. Permítanos presentarnos:

|    Luis Cortijo    |    Leandro Machaca    |    Enzo     | Jerimy Sandoval |
| ----------- | ----------- | ----------- | ----------- |
| ![](https://avatars.githubusercontent.com/u/84096868?v=4) | ![](https://avatars.githubusercontent.com/u/102132128?s=400&v=4) | ![](https://avatars.githubusercontent.com/u/90939274?v=4) | ![](https://avatars.githubusercontent.com/u/91238497?v=4) |
| [github.com/LuisEnriqueCortijoGonzales](https://github.com/LuisEnriqueCortijoGonzales) | [github.com/JLeandroJM](https://github.com/JLeandroJM) | [github.com/Enzoc30](https://github.com/Enzoc30) |  [github.com/Jerimy2021](https://github.com/Jerimy2021) |

---

## Preparación de Datos

Archivo: [`prepare_data.py`](prepare_data.py)

### Qué hace

-   **Valida imágenes**: descarta archivos corruptos o muy pequeños.
-   **Elimina duplicados**: usando hash SHA1 global con el fin de mantener el balance.
-   **Crea splits estratificados**: genera `train/`, `val/` y `test/` según proporciones definidas.
-   **Genera metadatos**: `class_index.json`, `classes.txt`, `dataset_stats.json`.

### Por qué así

-   **Calidad asegurada**: evitar imágenes rotas o irrelevantes mejora el entrenamiento.
-   **Generalización**: el split 70/15/15 asegura entrenamiento, validación y testeo equilibrados.
-   **Escalabilidad**: admite limitar imágenes por clase y elegir opciones dentro de sus argumentos.
-   **Flexibilidad**: permite copiar o mover archivos según necesidad.

---

## Entrenamiento del Modelo

Archivo: [`train.py`](train.py)

### Modelos soportados

-   **ResNet18 / ResNet50**
-   **MobileNetV3-Large**
-   **EfficientNet-B0**

Se usan pesos preentrenados en **ImageNet**, reemplazando la última capa por una adaptada a las clases de dinosaurios.

### Estrategia de entrenamiento

1. **Fase 1 (head training)**
    - Congela el backbone.
    - Entrena solo la capa final (cabeza).
    - Optimización rápida para adaptar a nuevas clases.
2. **Fase 2 (fine-tuning)**
    - Descongela toda la red.
    - Entrena con tasa de aprendizaje menor.
    - Ajusta todo el modelo a las particularidades del dataset.

### Mejoras aplicadas

-   **Data Augmentation fuerte**: crops, flips, jitter, perspectiva y rotaciones = más robustez.
-   **Mixup y CutMix**: combinaciones lineales o enmascaradas de imágenes para evitar overfitting.
-   **Label smoothing**: suaviza etiquetas para reducir sobreajuste.
-   **OneCycleLR**: scheduler dinámico que acelera convergencia.
-   **Early Stopping**: evita entrenamientos excesivos sin mejora.
-   **Weighted Sampler**: opcional, útil para datasets desbalanceados.

### Resultados guardados

-   Pesos: `model.pth` (mejor modelo por testing).
-   Métricas: `metrics.json`.
-   Historia de entrenamiento: `history.json`.
-   Matriz de confusión: `confusion_matrix.png`.

### Por qué así

-   **Dos fases**: acelera entrenamiento y evita mal ajuste inicial logrando ser ejecutado en local.
-   **Augmentations modernos**: generan robustez frente a variaciones del dataset.
-   **Mixup / CutMix**: Mejoran generalización en datasets pequeños/medianos.

---

## Frontend

Archivo: [`frontend.py`](frontend.py)

### Funcionalidades

-   **Carga de imágenes** vía interfaz Gradio.
-   **Predicciones Top-K** con probabilidades.
-   **GradioAPP**: Despliegue en la nube capaz de ser usado por cualquier persona con el link (mientras este activo el archivo).
-   **Configuración dinámica**: detección de arquitectura y clases desde el checkpoint.

### Por qué así

-   **Interpretabilidad**: Forma sencilla de ver los resultados.
-   **Usabilidad**: Gradio permite probar el modelo sin escribir código.
-   **Flexibilidad**: soporta múltiples arquitecturas (ResNet, MobileNet, EfficientNet).

---

## Cómo usar

### 1. Preparar dataset

```bash
python prepare_data.py --src_dir dinosaur_dataset --out_dir datasets/dinosaurs \
  --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --mode copy
```

### 2. Train

```bash
python train.py --data_dir datasets/dinosaurs --out_dir artifacts \
  --model resnet50 --img_size 256 \
  --epochs_head 5 --epochs_ft 20 \
  --lr_head 1e-3 --lr_ft 5e-4 --batch_size 32 \
  --mixup_alpha 0.2 --label_smoothing 0.1 --amp
```

### 3. Despliegue

```bash
python frontend.py
```

### 4. Disfruta
