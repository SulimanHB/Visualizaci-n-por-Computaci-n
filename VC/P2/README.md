# 🤖 Práctica 2 🤖

## Descripción del trabajo
En esta práctica seguimos explorando el uso de OpenCV, centrándonos en cuatro tareas:

- Cuenta de pixeles por filas
  
  ### 📏 Tarea 1 📏
  En la tarea 1 se trabaja con la imagen del mandril aplicando técnicas de detección de bordes y análisis de filas. El flujo de procesamiento es el siguiente:

  - Se carga la imagen y se convierte a escala de grises con `cv2.cvtColor`.
  - Se aplica el detector de bordes de **Canny** (`cv2.Canny`) para resaltar los contornos principales de la imagen.
  - A partir de la imagen binaria resultante, se realiza una **cuenta de píxeles blancos por fila**. Para ello:
    - Se usa `cv2.reduce` para obtener la suma de valores de cada fila.
    - Se normalizan los valores dividiendo entre 255, de manera que cada unidad corresponde a un píxel blanco.
  - Se identifica la fila con el **máximo número de píxeles blancos** mediante `np.argmax`.
  - Se calcula un umbral equivalente al **90% del valor máximo**, y se seleccionan las filas que lo superan.
  - Finalmente, se generan representaciones gráficas con **Matplotlib**:
    - En la primera figura se muestra la imagen binarizada con el detector de Canny.
    - En la segunda figura se presenta la **distribución de píxeles blancos por fila**, destacando:
      - El máximo de píxeles en color rojo.
      - El umbral del 90% en color naranja.
      - Las filas que superan dicho umbral marcadas en color verde.

  El resultado permite **visualizar y analizar qué filas concentran la mayor cantidad de píxeles blancos**, lo que facilita comprender la distribución de bordes detectados       en la imagen.


- Procesamiento y análisis de imágenes mediante umbralizado.

  ### 🖼️ Tarea 2 🖼️
  En la tarea 2 se trabaja con la conocida imagen del mandril. El flujo de procesamiento es el siguiente:

  - Se convierte la imagen a escala de grises con `cv2.cvtColor`.
  - Se suaviza con un filtro Gaussiano (`cv2.GaussianBlur`) para eliminar altas frecuencias.
  - Se aplica el operador Sobel en las direcciones **x** e **y** y se combinan los resultados (`cv2.add`).
  - Se convierte la imagen a 8 bits (`cv2.convertScaleAbs`) y se aplica un umbral binario con `cv2.threshold`.
  - Se realiza un conteo de píxeles no nulos en cada fila y columna usando `np.count_nonzero`.
  - Se calculan los máximos de filas y columnas, y se seleccionan aquellas que superan el **90%** del máximo.
  - Finalmente, se remarcan dichas filas (en rojo) y columnas (en verde) sobre la imagen original con `cv2.line`.

  Con todos estos pasos se obtiene como resultado una imagen en la que se muestran las filas y columnas con mayor cantidad de píxeles blancos.

- Demostración de cambio de modo de procesamiento en Webcam.

  ### 📷 Tarea 3 📷
  En la tarea 3 se propone crear un demostrador interactivo con la webcam, con varios modos de visualización:

  - **Modo 0 (Normal):** Muestra la cámara en tiempo real sin cambios.  
  - **Modo 1 (Inverso):** Aplica un negativo de la imagen (`cv2.bitwise_not`).  
  - **Modo 2 (Reducción de bits):** Se reduce la profundidad de color (cuantización) variando los bits por canal RGB.  

  📌 **Controles del demostrador**:
  - `m` → Cambiar entre modos.  
  - `a` → Reducir la cantidad de bits (mínimo 1).  
  - `d` → Aumentar la cantidad de bits (máximo 8).  
  - `q` → Salir del programa.  

  En el modo 1 se observa cómo la imagen se convierte en tonalidad negativa y, en el modo 2, la imagen muestra en la esquina superior izquierda el número de bits actuales. Esto permite apreciar cómo cambia la cantidad de colores posibles gracias a los bits y cómo la imagen se degrada visualmente al llegar hasta 1 bit como mínimo.

- Creación de un demostrador interactivo con la cámara.
  ### 🎮 Tarea 4 🎮
  En la tarea 4 se implementa un **demostrador interactivo de control musical mediante gestos de la mano**, combinando el uso de **MediaPipe** para el reconocimiento de gestos y **Pygame** para la reproducción de audio. El flujo de procesamiento es el siguiente:

  - Se inicializa el módulo `pygame.mixer` y se cargan varias pistas de audio (rock, reggae y surf).  
  - Se configura **MediaPipe Hands** con un umbral de confianza para la detección y el seguimiento de la mano.  
  - Se captura vídeo en tiempo real desde la cámara (`cv2.VideoCapture`).  
  - Para cada frame:
    - Se procesa la imagen en formato RGB con MediaPipe para detectar la mano y extraer sus **landmarks**.  
    - Se dibujan los puntos y conexiones de la mano sobre la imagen.  
    - Se aplica la función `detect_gesture`, que analiza la posición relativa de los dedos:
      - **Gesto rock** → `[0,1,0,0,1]` (índice y meñique extendidos).  
      - **Gesto reggae** → `[0,1,1,0,0]` (índice y medio en forma de “V”).  
      - **Gesto surf** → `[1,0,0,0,1]` (pulgar y meñique extendidos).  
  - Según el gesto detectado:
    - Si cambia respecto al anterior, se detiene la reproducción en curso.  
    - Si el gesto corresponde a uno válido, se carga la canción asociada y se reproduce en bucle.  
  - El resultado es un **control musical manos libres**, donde el usuario puede alternar entre estilos musicales simplemente mostrando gestos específicos frente a la cámara.

  Este demostrador pone en práctica la integración de **visión por computador (MediaPipe)** con la **interactividad multimedia (Pygame)**, mostrando un ejemplo atractivo y creativo de interacción natural basada en gestos.


## Autoría
Este trabajo ha sido realizado por:  

**Pablo Medina Quintana:** Tareas 1 y 4  
**Suliman Hassan:** Tareas 2 y 3

## Fuentes y referencias
Durante el desarrollo de la práctica se han consultado y utilizado las siguientes fuentes:  

- Documentación oficial de [NumPy](https://numpy.org/doc/).  
- Documentación oficial de [OpenCV](https://docs.opencv.org/).  
- Documentación oficial de [MediaPipe](https://developers.google.com/mediapipe).  
- Documentación oficial de [Pillow](https://pillow.readthedocs.io/).  
- Documentación oficial de [Matplotlib](https://matplotlib.org/stable/contents.html).  
- Documentación oficial de [Pygame](https://www.pygame.org/docs/).  

## Requisitos de instalación
Para ejecutar el cuaderno correctamente es necesario tener instalado **Python 3.8 o superior** y las siguientes librerías:  

```bash
pip install numpy opencv-python pillow matplotlib mediapipe pygame



