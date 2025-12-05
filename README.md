# BreastPathQ Solution - Tumor Cellularity Quantification

Solución completa para el desafío BreastPathQ utilizando Deep Learning con Transfer Learning (ResNet50).

## 📊 Resultados

- **Métrica PK (Pearson Correlation):** 0.8208
- **R² Score:** 0.6420
- **MAE:** 0.1233
- **RMSE:** 0.1764
- **Loss (MSE) - Train:** 0.0258
- **Loss (MSE) - Validation:** 0.0311

## 🏗️ Arquitectura del Modelo

- **Base:** ResNet50 pre-entrenado en ImageNet
- **Capas congeladas:** 80 (de 175 totales)
- **Capas entrenables:** 95
- **Head personalizado:** Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.2) → Dense(128) → Dropout(0.15) → Dense(1)
- **Parámetros totales:** ~25M

## 🔧 Configuración

### Hiperparámetros
- **Image Size:** 224x224
- **Batch Size:** 32
- **Epochs:** 50
- **Learning Rate:** 0.00005
- **Optimizer:** Adam con ReduceLROnPlateau
- **Loss Function:** Mean Squared Error (MSE)

### Data Augmentation
- Rotación: ±20°
- Flips horizontales y verticales
- Zoom: ±10%
- Ajuste de brillo: ±20%

## 📁 Estructura del Proyecto

```
Proyecto/
├── modelo.ipynb              # Notebook principal con todo el pipeline
├── proyecto.md               # Descripción del proyecto
├── breastpathq_submission.csv # Predicciones para test set
├── best_model_resnet50.keras  # Modelo entrenado (no incluido en repo)
├── breastpathq/
│   ├── datasets/
│   │   ├── train_labels.csv
│   │   ├── train/           # Imágenes de entrenamiento (no incluidas)
│   │   ├── validation/      # Imágenes de validación (no incluidas)
│   │   └── cells/           # Archivos XML con anotaciones
│   └── submission/
└── breastpathq-test/
    ├── test_patches/         # Imágenes de test (no incluidas)
    ├── test_patient_ids.csv
    └── val_labels.csv
```

## 📥 Datos del Proyecto

**IMPORTANTE:** Las imágenes del dataset no están incluidas en este repositorio debido a su tamaño (>2GB).

Para reproducir este proyecto:

1. Descarga el dataset desde: https://breastpathq.grand-challenge.org/
2. Coloca las imágenes en las carpetas correspondientes:
   - `breastpathq/datasets/train/`
   - `breastpathq/datasets/validation/`
   - `breastpathq-test/test_patches/`

## 🚀 Uso

1. **Instalar dependencias:**
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn pillow opencv-python scikit-learn scipy
```

2. **Ejecutar el notebook:**
```bash
jupyter notebook modelo.ipynb
```

3. **Secciones del notebook:**
   - **A:** Descripción del problema
   - **B:** Carga de datos
   - **C:** Análisis exploratorio
   - **D:** Construcción y entrenamiento del modelo
   - **E:** Evaluación y métricas
   - **F:** Conclusiones
   - **G:** Predicciones en test set

## 📈 Características Destacadas

- ✅ Transfer Learning con ResNet50
- ✅ Data Augmentation completo
- ✅ Early Stopping y ReduceLROnPlateau
- ✅ Análisis de overfitting/underfitting automatizado
- ✅ Visualizaciones comprehensivas
- ✅ Métricas múltiples (PK, R², MAE, RMSE)
- ✅ Documentación completa en español

## 👥 Autores

Proyecto final para el curso "Introducción a los Sistemas Inteligentes" - Universidad Nacional de Colombia (UNAL)

## 📅 Fecha

Diciembre 2025

## 📄 Licencia

Este proyecto es parte de un trabajo académico.

## 🔗 Enlaces

- [Competition Website](https://breastpathq.grand-challenge.org/)
- [Dataset Paper](https://www.nature.com/articles/s41597-019-0290-4)
