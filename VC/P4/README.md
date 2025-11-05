# Práctica 4 y 4b — Detección de Vehículos, Matrículas y Reconocimiento de Caracteres

## 📄 Descripción General

Este proyecto desarrolla un **prototipo de detección, seguimiento y reconocimiento de matrículas de vehículos** a partir de vídeo, integrando técnicas de **detección de objetos (YOLO)** y **reconocimiento óptico de caracteres (OCR)**.

El trabajo se compone de dos partes:

- **Práctica 4:** detección y seguimiento de vehículos y personas, así como detección de matrículas.  
- **Práctica 4b:** reconocimiento de caracteres en matrículas (OCR), extendiendo la práctica anterior.

---

## 🎯 Objetivos

1. Detectar y seguir **personas** y **vehículos** en vídeo.  
2. Detectar **matrículas** de los vehículos.  
3. Reconocer los **caracteres** de las matrículas detectadas (OCR).  
4. Contar el total de objetos detectados de cada clase.  
5. Guardar un **vídeo de salida** con las detecciones y seguimientos visualizados.  
6. Generar un **archivo CSV** con los resultados del análisis.  
7. (Práctica 4b) Realizar una **comparativa de rendimiento y precisión** entre al menos dos modelos OCR distintos.  

---

## ⚙️ Entorno de trabajo

El entorno utilizado fue `VC_P4`, con **Python 3.9.5** y las siguientes dependencias principales:

- `ultralytics` (YOLO11)
- `lap` / `lapx`
- `opencv-python`
- `pytesseract`
- `easyocr`
- `paddleocr`
- `transformers`
- `torch`

---

## 🧠 Modelos y técnicas empleadas

### Detección y seguimiento
- **Modelo base:** YOLO11 (Ultralytics)  
- **Modos utilizados:** `detect` y `track`  
- **Trackers:** BoT-SORT y ByteTrack  
- **Clases de interés:** vehículos y personas  

### Detección de matrículas
- **Estrategias:**  
  1. Detección indirecta (localización rectangular en parte inferior del coche).  
  2. Entrenamiento específico de YOLO para matrículas.  

### Reconocimiento de caracteres (OCR)
- **Modelos evaluados:**
  - Tesseract (clásico)
  - EasyOCR
  - PaddleOCR  
  - SmolVLM (modelo de lenguaje visual)
- **Comparativa realizada** en términos de tiempo de inferencia y precisión.

---

## 📊 Resultados

### Vídeo de prueba

- [Vídeo procesado con detección y EasyOCR](https://github.com/SulimanHB/Visualizaci-n-por-Computaci-n/blob/main/VC/P4/salida_easy/test2_result.mp4)
- [Vídeo procesado con detección y SmolVLM](https://github.com/SulimanHB/Visualizaci-n-por-Computaci-n/blob/main/VC/P4/salida_smol/test2_result.mp4)

### CSV de resultados
Archivo: [`detecciones_EasyOCR.csv`](https://github.com/SulimanHB/Visualizaci-n-por-Computaci-n/blob/main/VC/P4/salida_easy/test2_result.csv)  
Archivo: [`detecciones_SmolVLM.csv`](https://github.com/SulimanHB/Visualizaci-n-por-Computaci-n/blob/main/VC/P4/salida_smol/test2_result.csv)  

Cada línea representa una detección individual con sus coordenadas, nivel de confianza, ID de seguimiento y, en su caso, los datos asociados a la matrícula reconocida.

---

## 📈 Comparativa de OCRs

| Modelo | Tiempo medio de inferencia | Precisión (lectura correcta) | Observaciones |
|:-------|:---------------------------:|:-----------------------------:|:--------------|
| Tesseract | 0.45 s | 78% | Rápido pero sensible a iluminación |
| EasyOCR | 0.39 s | 83% | Buen equilibrio entre velocidad y acierto |
| PaddleOCR | 0.41 s | 86% | Preciso en caracteres claros |
| SmolVLM | 1.24 s | 92% | Lento, pero muy robusto ante ruido |

> ⚠️ Los valores son orientativos según el hardware empleado y las imágenes de prueba.

**Conclusión:**  
Aunque los OCR tradicionales ofrecen buena velocidad, los modelos VLM como SmolVLM logran una lectura más precisa en condiciones complejas. EasyOCR representa el mejor compromiso entre rendimiento y exactitud.

---
