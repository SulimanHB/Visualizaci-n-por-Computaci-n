# **Práctica 5 – Virtualización por Computador**  
## **Extracción de Información Facial y Prototipos Interactivos**

Este repositorio contiene el desarrollo de dos prototipos interactivos basados en la extracción de información facial mediante técnicas de *Computer Vision* y *Deep Learning*. La práctica se divide en dos partes:

1. **Prototipo libre** que reacciona a gestos de la mano y aplica filtros faciales + música.  
2. **Prototipo que emplea un modelo entrenado por nosotros** para extraer información biométrica (emociones faciales) y generar respuestas visuales.

Ambos prototipos utilizan la webcam en tiempo real y generan una reacción visual y/o sonora dependiendo de las detecciones realizadas.

---

# █ **Prototipo 1 – “Mano Musical con Filtros Faciales”**  
### **(Prototipo de temática libre)**

Este prototipo combina **detección de manos**, **reconocimiento de gestos**, **filtros visuales sobre el rostro** y **reproducción musical automática**.  
Está inspirado en experiencias creativas tipo Snapchat/Instagram, pero con control por gestos.

### ✔ **Tecnologías empleadas**
- **MediaPipe Hands** – para detectar landmarks de la mano.  
- **MediaPipe FaceMesh** – para colocar filtros faciales alineados.  
- **OpenCV** – para vídeo, filtros de color y superposición gráfica.  
- **Pygame** – para reproducción de música según el gesto detectado.

### ✔ **Gestos soportados y reacciones**
Cada gesto corresponde a un estilo musical y un filtro facial:

| Gesto detectado | Interpretación | Acción visual                 | Música reproducida |
|-----------------|----------------|-------------------------------|---------------------|
| 🤘 Rock Sign    | rock           | Tinte rojo + filtro rock      | acdc.mp3            |
| ✌️ Two-Fingers   | reggae         | Tinte verde + filtro rasta    | bob_marley.mp3      |
| 🤙 Hang Loose   | surf           | Tinte azul + filtro surf      | surf.mp3            |

### ✔ **Funcionamiento**
1. El usuario hace un gesto delante de la cámara.  
2. Se detecta su forma según la posición de los landmarks de la mano.  
3. Se reproduce música automáticamente según el gesto detectado.  
4. Se aplica un tinte de color global y un filtro PNG alineado a los ojos mediante FaceMesh.

Este prototipo crea una experiencia visual divertida e interactiva, integrando manos, rostro y sonido.

---

# █ **Prototipo 2 – “Detector de Emociones con SVM + MobileNet”**  
### **(Prototipo obligatorio con modelo entrenado por el estudiante)**

En este prototipo se entrena un modelo personalizado para la **clasificación de emociones faciales**, usando embeddings generados con MobileNet V2 y un clasificador **SVM** entrenado desde cero.

YOLO se utiliza para detectar los rostros en tiempo real.

### ✔ **Emociones detectadas**
- angry  
- disgust  
- fear  
- happy  
- sad  
- surprise  
- neutral  

### ✔ **Proceso de entrenamiento**
1. Se cargan las imágenes del dataset organizado por clases.  
2. Se extrae un embedding de 1280 dimensiones con MobileNet V2 (sin la capa final).  
3. Se equilibra el dataset mediante submuestreo por clase.  
4. Se entrena un **SVM con kernel RBF** usando un pipeline con `StandardScaler`.  
5. Se guarda el modelo final en `emotion_svm_mobilenet.pkl`.

### ✔ **Funcionamiento en tiempo real**
- YOLO detecta el rostro.  
- MobileNet obtiene el embedding.  
- El SVM calcula las probabilidades de cada emoción.  
- Se colorea toda la imagen con un overlay asociado a la emoción detectada.  
- Se usa un historial para suavizar fluctuaciones rápidas entre emociones.

### ✔ **Colores por emoción**
| Emoción   | Color dominante |
|-----------|----------------|
| angry     | rojo           |
| disgust   | verde oscuro   |
| happy     | amarillo       |
| sad       | naranja apagado|
| surprise  | magenta        |
| neutral   | gris           |

---

# 🎬 **Vídeos / GIF de demostración**

### Prototipo 1 – Mano Musical
![Prototipo 1](Tarea1VC.gif)

### Prototipo 2 – Detector de Emociones
![Prototipo 2](emotion_capture.gif)

---

# 🔧 **Requisitos**
- Python 3.8+  
- OpenCV  
- MediaPipe  
- PyTorch + Torchvision  
- Ultralytics YOLO  
- scikit-learn  
- joblib  
- pygame  

---


