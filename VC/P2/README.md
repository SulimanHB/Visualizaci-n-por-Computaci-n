🤖 Práctica 2 🤖

En esta práctica seguimos explorando el uso de OpenCV, centrándonos en:

- Cuenta de pixeles por filas

- Procesamiento y análisis de imágenes mediante umbralizado.

- Demostración de cambio de modo de procesamiento en Webcam.

- Creación de un demostrador interactivo con la cámara.


Tarea 1



🖼️ Tarea 2 🖼️

En la tarea 2 se trabaja con la conocida imagen del mandril. El flujo de procesamiento es el siguiente:

- Se convierte la imagen a escala de grises con cv2.cvtColor.

- Se suaviza con un filtro Gaussiano (cv2.GaussianBlur) para eliminar altas frecuencias.

- Se aplica el operador Sobel en las direcciones x e y y se combinan los resultados (cv2.add).

- Se convierte la imagen a 8 bits (cv2.convertScaleAbs) y se aplica un umbral binario con cv2.threshold.

- Se realiza un conteo de píxeles no nulos en cada fila y columna usando np.count_nonzero.

- Se calculan los máximos de filas y columnas, y se seleccionan aquellas que superan el 90% del máximo.

- Finalmente, se remarcan dichas filas (en rojo) y columnas (en verde) sobre la imagen original con cv2.line.

Con todos estos pasos nos da como resultado una imagen en el que se muestran las filas y columnas con más pixeles blancos.




📷 Tarea 3 📷

En la tarea 3 se propone crear un demostrador interactivo con la webcam, con varios modos de visualización:

Modo 0 (Normal): Muestra la cámara en tiempo real sin cambios.

Modo 1 (Inverso): Aplica un negativo de la imagen (cv2.bitwise_not).

Modo 2 (Reducción de bits): Se reduce la profundidad de color (cuantización) variando los bits por canal RGB.

📌 Controles del demostrador:

- m → Cambiar entre modos.

- a → Reducir la cantidad de bits (mínimo 1).

- d → Aumentar la cantidad de bits (máximo 8).

- q → Salir del programa.

En el modo 1 vemos como la imagen se convierte en tonalidad negativa y en el modo 2, la imagen muestra en la esquina superior izquierda el número de bits actuales, lo que permite observar cómo cambia la cantidad de colores posibles gracias a los bits y cómo se degrada visualmente la imagen llegando hasta 1 bit como minimo.



Tarea 4


