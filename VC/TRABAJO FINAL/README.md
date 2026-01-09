# Graffiti Virtual AR - Proyecto de Visión por Computador

Este proyecto implementa una aplicación de **Realidad Aumentada (RA)** que permite a los usuarios dibujar virtualmente sobre superficies del mundo real utilizando gestos manuales. El sistema integra **MediaPipe** para el seguimiento de manos, **OpenCV** para el cálculo de homografías y persistencia visual, y **Stable Diffusion (IA Generativa)** para transformar bocetos a mano alzada en arte estilo graffiti profesional en tiempo real.

## 📋 Características Principales

* **Seguimiento de Manos (Hand Tracking):** Dibujo natural utilizando el dedo índice.
* **Control por Gestos:**
    * *Pinza (Pinch):* Índice + Pulgar para dibujar.
    * *Mano Abierta:* Borrador total del lienzo.
    * *Swipe Vertical:* Índice + Medio para cambiar el grosor del trazo.
    * *Swipe Horizontal:* Índice solo para cambiar el color.
* **Persistencia AR:** Utiliza el algoritmo **ORB** (Oriented FAST and Rotated BRIEF) para detectar características en la pared y mantener el dibujo "pegado" a la superficie aunque la cámara se mueva.
* **IA Generativa en Local:** Integración de **Stable Diffusion v1.5** con **ControlNet Scribble** y **LCM LoRA** (Modo Turbo) para generar graffitis de alta calidad en menos de 2 segundos.

## 🛠️ Requisitos del Sistema

### Hardware
* **GPU:** Tarjeta gráfica NVIDIA (Recomendado RTX 3060 o superior) con soporte CUDA.
* **Cámara:** Webcam estándar o Smartphone (vía Iriun Webcam).

### Dependencias de Software
Es necesario tener **Python 3.10+** instalado. Instala las librerías necesarias ejecutando:

```bash
pip install opencv-python opencv-contrib-python mediapipe numpy torch torchvision diffusers transformers accelerate peft
```


## 📱 Configuración de Iriun Webcam (Cámara del Móvil)

Para utilizar la cámara de tu smartphone como fuente de entrada de alta calidad:

### 🔧 Pasos de Instalación
1. **Instalar App:** Descarga **Iriun Webcam** en tu móvil (iOS/Android).
2. **Instalar Drivers:** Descarga e instala **Iriun Webcam for Windows** en tu PC desde [https://iriun.com](https://iriun.com).
3. **Conectar:** Asegúrate de que ambos dispositivos estén en la misma red Wi-Fi o conectados por cable USB (recomendado para menor latencia).
4. **Verificar:** Abre la aplicación de Iriun en el PC. Deberías ver la imagen de tu móvil.

### ⚙️ Configuración en Código
En el archivo `main.py`, la línea:

```python
cap = cv2.VideoCapture(1) #selecciona el índice de la cámara.
```
Si el script no abre la cámara de Iriun, prueba cambiando el 1 por 0 o 2.

## 🚀 Instalación y Ejecución
1. Clona este repositorio o descarga los archivos en tu equipo
2. Asegúrate de que los archivos main.py y improve_IA.py estén en la misma carpeta.
3. Ejecuta el script principal

## Guía de usuario 
- **Definir superficie:** Haz clic con el ratón en 4 puntos de la pantalla sobre la pared
**(Esquina Superior-Izq → Superior-Der → Inferior-Der → Inferior-Izq)** para delimitar el área de trabajo.
. **Resetear Puntos:** Pulsa la tecla `R` para volver a seleccionar los 4 puntos de la pared.
- **Dibujar:** Acerca tu mano a la cámara. Haz el gesto de pinza (juntar índice y pulgar) para empezar a pintar.
- **Transformación IA:** Pulsa la tecla `I`. El dibujo se congelará, será procesado por la IA y devuelto como un graffiti realista.
- **Guardar Graffiti:** Pulsa la tecla `S` para guardar el dibujo actual en el mundo AR (persistencia). Esto limpia el lienzo para dibujar uno nuevo.
- **Salir:** Pulsa la tecla `ESC` para cerrar la aplicación.


---

**Asignatura:** Visión por Computador

**Autores:** Carlos Falcón Castellano, Suliman Hassan El Boutaybi y Pablo Medina Quintana

