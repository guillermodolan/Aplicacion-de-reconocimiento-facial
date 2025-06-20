# Face Recognition App

Esta es una aplicación desarrollada como parte de un proyecto para la asignatura "Soporte a la Gestión de Datos con programación Visual" en la Universidad Tecnológica Nacional, Facultad Regional Rosario. Desarrollada desde Septiembre hasta Octubre del 2023, la aplicación utiliza Python y la biblioteca Kivy para crear una interfaz gráfica interactiva que permite a los usuarios realizar acciones basadas en gestos de manos.

## Características principales

- **Inicio de Sesión**: Los usuarios pueden iniciar sesión utilizando la biblioteca DeepFace para la autenticación facial.

- **Grabación de Video y captura de foto**: Cuando un usuario se encuentra logueado, la aplicación puede detectar la mano derecha utilizando la biblioteca MediaPipe y comenzar a grabar video hasta que la mano ya no sea detectada. En cambio, si se detecta la mano izquierda, la aplicación activa un temporizador de 5 segundos y toma una foto.

- **Galería de archivos**: Los usuarios pueden ver un listado de fotos y videos previamente capturados y reproducirlos.

Las tecnologías utilizadas en dicha aplicación son:  
**Back-End**: Python
**Front-End**: Kivy (Python Framework)

## Requisitos

Previa a la ejecución de esta aplicación, es necesario tener instalados los siguientes requisitos previos:
- La biblioteca de Pyhton "DeepFace".
- El framework "MediaPipe".
- El módulo de Python "EasyGUI".
