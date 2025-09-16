# 🤖 **Práctica 1** 🤖

En esta práctica veremos los primeros pasos hacia el uso de OpenVC, con tareas como:

- Tablero de Ajedrez
- Modrian
- Modificación de video
- Detector del pixel mas claro y oscuro
- Pop Art

## ♕ **Tarea 1** ♕

En la tarea 1, se partirá de un tablero de ancho y alto de 800 pixeles, en el que el tablero seria un 8x8, teniendo por cuadrado 100x100 pixeles.

Posteriormente, definido ya el tablero, se crearia el cuadrado, en el cual, para definir el pixel blanco y negro, se usa una variable llamada tablero, en el que nos dara un resultado 0 o 1, si es 0 blanco, al reves negro.


## 🖼️​ **Tarea 2** 🖼️​

En la tarea 2, se propone la creación de una imagen con el estilo ***Mondrian***, esto se logra gracias a las herramientas que nos proporciona la libreria de **CV2**, ya que con esta, se puede usar la función *rectangle* y *line* para lograr un arte similar al ***Mondrian***, esto se hace seleccionando las coordenadas (x1,y1) y (x2,y2), por ejemplo en este caso *cv2.rectangle(T2, (0, 0), (300, 300), (0, 0, 255), -1)* y *cv2.line(T2, (300, 0), (300, 800), (0, 0, 0), 10)*
- T2 seria la imagen creada, siendo 800x800.
- Las coordenadas (0,0) serian las iniciales, de donde empieza, hasta donde llega (300,300), en el caso del rectangulo. 
- Las lineas funcionarian igual, seleccionando un punto de origen y un punto de destino.
- Para los colores, se van alternando valores entre 0-255, por ejemplo(0,0,255) o (145,38,65).


## 📷​​ **Tarea 3** 📷​

En la tarea 3, se propone modificar uno de los 3 planos RGB de la camara, una modificacion interesante seria el color negativo, tornando el verde al negativo, esto se logra con la variable ***g***, ya que con su valor habra que restarle al 255, dando asi la tonalidad negativa, guardando este valor en ***gn***, por ultimo seria en la variable **collage**, añadir r, gn y b, siendo r(rojo) gn(green negative) y b(blue).


## 🔴​​ **Tarea 4** 🔵​

En la tarea 4, primero hacemos uso de la función de ratón, para comprobar los colores RGB​.

Luego con la variable gray y cv2.cvtColor, transformamos la imagen a escala de grises, para trabajar con los niveles de brillo, posteriormente, mostramos los circulos con **cv2.circle** con *minLoc* para buscar el pixel más oscuro y *maxLoc* para buscar el pixel más blanco.

En otro lado, para buscar en una zona rectangular 8x8, no solo es buscar el pixel, si no obtener un promedio de la zona. Primero se pasa la imagen a escalado de grises y se reduce con ***cv2.resize***, asi, cada bloque queda representado por un valor promedio, dibujando los rectangulos mediante ***cv2.rectangle***.

Con esto, se mostrarán circulos mostrando el pixel más oscuro(azul) y más clara(rojo), y para las zonas, mediantes rectangulos.


## 🍭​​​ **Tarea 5** 🍭​

En la tarea 5, con la variable *cam*, redimensionaremos el frame aún tamaño más pequeño, para luego dentro del collage, montar 9 imagenes, con las variables img*, aplicando un filtro diferente en cada uno, gracias a las herramientas que da ***cv2*** con funciones como **COLORMAP** o **cam[:, :, [x,x,x]]**.

Una vez aplicado el filtro en cada img, se ajuntaran en un collage 3x3, teniendo:

- TOP: La parte de arriba.
- MID: La parte del medio.
- BOTTOM: La parte de abajo.


