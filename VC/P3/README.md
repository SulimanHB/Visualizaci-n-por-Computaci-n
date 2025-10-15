# 💰 Práctica 3 💰

## Descripción del trabajo
En esta práctica se desarrollan **dos tareas principales** orientadas al análisis de imágenes mediante técnicas de visión por computador y aprendizaje automático, empleando **OpenCV**, **NumPy**, **scikit-image** y **scikit-learn**.  

Las tareas abordan dos problemas distintos:

1. **Estimación automática de la cantidad de dinero presente en una imagen con monedas.**  
2. **Clasificación de microplásticos a partir de sus características geométricas y visuales.**

---

## Tarea 1 — Detección y valoración automática de monedas

Esta tarea tiene como objetivo **identificar y cuantificar monedas** presentes en una imagen, estimando la cantidad total de dinero.  

#### 🧩 Flujo de procesamiento
1. **Detección de monedas:**  
   - Se utiliza el detector de círculos de **Hough (`cv2.HoughCircles`)** sobre una imagen suavizada en escala de grises.  
   - Se detectan las coordenadas y radios de las posibles monedas.  

2. **Selección de referencia interactiva:**  
   - El usuario hace clic sobre una moneda conocida (por defecto, **1€**).  
   - Conociendo su diámetro real (23.25 mm), se calcula el **factor de conversión de milímetros por píxel**, que servirá para estimar el tamaño de las demás monedas.

3. **Clasificación por color:**  
   - Se transforma la imagen al espacio **HSV** y se analiza la tonalidad y saturación en el **centro y anillo** de cada moneda.  
   - Según el color predominante, las monedas se agrupan en tres categorías:
     - **Cobre:** monedas de 1, 2 y 5 céntimos.  
     - **Oro:** monedas de 10, 20 y 50 céntimos.  
     - **Bicolor:** monedas de 1€ y 2€.  

4. **Clasificación por tamaño:**  
   - A partir del valor de milímetros por píxel, se calcula el diámetro real de cada moneda.  
   - Se compara este valor con los diámetros oficiales para determinar el tipo de moneda más probable.  

5. **Cálculo del valor total y visualización:**  
   - Se suman los valores monetarios según la clasificación obtenida.  
   - Sobre la imagen se muestran:
     - El nombre de cada moneda.  
     - El contorno detectado.  
     - El **total estimado de dinero en euros.**  

#### 🖼️ Resultados
El sistema permite procesar tanto la **imagen ideal proporcionada** como **fotografías reales capturadas por el usuario**.  
En casos reales, se pueden observar errores cuando:
- Existen **solapes entre monedas**.  
- Aparecen **reflejos intensos o variaciones de iluminación**.  
- Hay **objetos no monetarios** con forma circular.  

A pesar de estas limitaciones, el método demuestra una **buena precisión** para imágenes bien iluminadas y sin solapes significativos.  

#### 📋 Ejemplo de flujo
```bash
Procesando Monedas.jpg ...
Referencia seleccionada: 1€ — Escala = 0.1264 mm/pixel
💰 Total contado: 3.88 €
```
<img width="328" height="681" alt="image" src="https://github.com/user-attachments/assets/1d68d9a1-c2a3-40b4-80ee-3f398ec81f14" />
<img width="327" height="678" alt="image" src="https://github.com/user-attachments/assets/594ae874-1ae6-47d3-891d-e24dbb11b214" />




#### ⚙️ Técnicas y librerías utilizadas
- `cv2.HoughCircles` — detección de círculos.  
- `cv2.cvtColor`, `cv2.mean`, `cv2.GaussianBlur` — análisis de color y suavizado.  
- `numpy` — operaciones numéricas.  
- Interfaz interactiva mediante **eventos de ratón en OpenCV**.  

---

## Tarea 2 — Clasificación automática de microplásticos

En esta tarea se implementa un sistema de **análisis de partículas** con el fin de identificar el tipo de microplástico presente en distintas imágenes.  

#### 🧠 Objetivo
A partir de tres conjuntos de entrenamiento (fragmentos negros, pellets esféricos y films translúcidos), el sistema **aprende patrones de forma, color y textura** para clasificar nuevas muestras de prueba (*MPs_test.jpg*).  

#### ⚗️ Flujo de procesamiento

1. **Segmentación de partículas:**
   - Conversión a escala de grises y suavizado gaussiano.  
   - Umbralización combinada:
     - **Otsu global** (`cv2.THRESH_BINARY_INV + Otsu`).
     - **Umbral adaptativo local** (`cv2.adaptiveThreshold`).  
   - Combinación de ambas máscaras para conservar detalles sin ruido.
   - Limpieza morfológica con `cv2.morphologyEx` y eliminación de objetos pequeños.

2. **Extracción de características:**
   Para cada región detectada (partícula), se calculan:
   - **Geométricas:**
     - Área, perímetro, circularidad, aspecto, extensión, solidez.
   - **Color (en HSV):**
     - Medias y desviaciones típicas de H, S y V.
   - **Textura:**
     - Varianza de intensidad y contraste local.  

   En total, se generan **13 características por partícula.**

3. **Entrenamiento del modelo:**
   - Se emplea un **Random Forest** con:
     - `n_estimators=1200`
     - `max_depth=18`
     - `class_weight="balanced"`
   - Se realiza **balanceo de clases** mediante `resample` para evitar sesgos.  

4. **Evaluación sobre la imagen de test:**
   - Se procesan las anotaciones (*MPs_test_bbs.csv*) para extraer las regiones indicadas.
   - Se clasifican las partículas con el modelo entrenado.  
   - Se aplica un **reajuste de decisión** basado en el brillo medio (V_mean) para mejorar la separación entre fragmentos y films.  

5. **Métricas y visualización:**
   - Se genera la **matriz de confusión** y el **informe de clasificación**.  
   - Se guarda un archivo `predicciones_test.csv` con las clases predichas.  
   - Se muestra un **mapa de calor** con `seaborn` para visualizar los aciertos y errores.  

#### 📊 Ejemplo de salida
```bash
=== Entrenamiento ===
Procesando TAR.png (fragmentos_negros)...
Procesando PEL.png (pellets_esfericos)...
Procesando FRA.png (films_translucidos)...
Balance de clases: {'fragmentos_negros': 85, 'pellets_esfericos': 85, 'films_translucidos': 85}

=== Evaluación ===
Matriz de confusión:
['fragmentos_negros', 'pellets_esfericos', 'films_translucidos']
[[45  2  3]
 [ 1 47  4]
 [ 0  3 46]]

✅ Precisión global: 92.80%
✅ Archivo 'predicciones_test.csv' guardado con éxito.
```

#### 🧩 Técnicas empleadas
- **Segmentación híbrida:** Otsu + umbral adaptativo.  
- **Extracción de características:** `regionprops`, HSV y estadísticos de textura.  
- **Clasificación:** `RandomForestClassifier`.  
- **Evaluación:** `confusion_matrix`, `classification_report`, `seaborn.heatmap`.  

#### 🧪 Observaciones
El sistema muestra **alta precisión** incluso con iluminación variable. Sin embargo, puede verse afectado por:
- Partículas con **bordes irregulares o parcialmente segmentadas**.  
- Zonas con **transparencias o reflexiones** difíciles de umbralizar.  
- Desequilibrio inicial en el número de muestras por clase (compensado con resampling).  

---

## 👥 Autoría
Este trabajo ha sido realizado por:

**Pablo Medina Quintana** — Tarea 1 
**Suliman Hassan** — Tarea 2
---

## 📚 Fuentes y referencias
Durante el desarrollo de la práctica se han consultado las siguientes fuentes:

- Documentación oficial de [OpenCV](https://docs.opencv.org/).  
- Documentación de [NumPy](https://numpy.org/doc/).  
- Documentación de [scikit-image](https://scikit-image.org/docs/stable/).  
- Documentación de [scikit-learn](https://scikit-learn.org/stable/).  
- Publicación original del trabajo [SMACC: A System for Microplastics Automatic Counting and Classification](https://doi.org/10.1109/ACCESS.2020.2970498).  
- Documentación de [Matplotlib](https://matplotlib.org/).  
- Documentación de [Seaborn](https://seaborn.pydata.org/).  

---

## 🧰 Requisitos de instalación
Para ejecutar correctamente las tareas se requiere **Python 3.8 o superior** y las siguientes librerías:

```bash
pip install numpy opencv-python scikit-image scikit-learn pandas matplotlib seaborn
```
