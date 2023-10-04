import os
from os.path import isfile, join

from kivy.app import App
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
from deepface import DeepFace
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
import mediapipe as mp
import time
from kivy.uix.scrollview import ScrollView
from kivy.uix.popup import Popup
from kivy.uix.image import Image, AsyncImage
from kivy.uix.videoplayer import VideoPlayer


class ArchivoScreen(Screen):

    # Método para capturar el nombre de la foto, de la persona que se haya logueado. Esto se
    # asignará al botón almacenado en la pantalla de 'archivo'

    # Ejemplo: 'Guillermo.jpg', extrae solo el nombre Guillermo
    def get_button_text(self):
        # Verificar si la carpeta 'fotos' ya existe
        carpeta_fotos = "fotos"
        if not os.path.exists(carpeta_fotos):
            # Si no existe, crear la carpeta
            os.makedirs(carpeta_fotos)

        # Con esto capturamos el nombre de la foto, de la persona que se haya logueado.
        # Ejemplo: 'Guillermo.jpg', extraemos solo el nombre "Guillermo"
        for nombre_archivo in os.listdir(carpeta_fotos):
            if nombre_archivo.endswith(".jpg"):
                nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
                return nombre_sin_extension  # Devuelve el nombre sin extensión

    # Método que sirve para mostrar las imágenes.
    def mostrar_archivos(self):
        nombre = self.get_button_text()
        carpeta_fotos_y_videos = f"fotos_y_videos/{nombre}.jpg"

        # Verificar si la carpeta 'carpeta_fotos_y_videos' existe
        if not os.path.exists(carpeta_fotos_y_videos):
            print('No existe la carpeta')
        else:
            # Obtener una lista de archivos en la carpeta
            archivos = [f for f in os.listdir(carpeta_fotos_y_videos) if isfile(join(carpeta_fotos_y_videos, f))]

            # Limpiar el contenido anterior del ScrollView
            self.ids.scroll_view.clear_widgets()

            # Crear un layout para el ScrollView
            layout = BoxLayout(orientation='vertical', spacing=10)

            # Verificar si hay archivos en la carpeta
            if archivos:
                # Agregar cada archivo como un botón al layout
                for archivo in archivos:
                    btn = Button(text=archivo, size_hint_y=None, height=40)
                    if any(ext in archivo.lower() for ext in ('.png', '.jpg', '.jpeg', '.gif')):
                        # Si es una imagen, mostrar la imagen
                        btn.bind(on_press=lambda instance, file=archivo: self.mostrar_imagen(file))
                    elif any(ext in archivo.lower() for ext in ('.mp4', '.avi', '.mov', '.mkv')):
                        # Si es un video, mostrar el video
                        btn.bind(on_press=lambda instance, file=archivo: self.mostrar_video(file))
                    layout.add_widget(btn)
            else:
                # Si no hay archivos, mostrar un mensaje
                layout.add_widget(Label(text="No hay archivos en la carpeta."))

            # Agregar el layout al ScrollView
            self.ids.scroll_view.add_widget(layout)

    def mostrar_imagen(self, archivo):
        # Capturo el nombre de la persona logueada
        nombre = self.get_button_text()

        ruta_imagen = join(f"fotos_y_videos/{nombre}.jpg", archivo)

        # Mostrar la imagen en un Popup
        img = AsyncImage(source=ruta_imagen)
        popup = Popup(title=archivo, content=img, size_hint=(None, None), size=(400, 400))
        popup.open()

    def mostrar_video(self, archivo):
        # Capturo el nombre de la persona logueada
        nombre = self.get_button_text()

        ruta_video = join(f"fotos_y_videos/{nombre}.jpg", archivo)

        # Mostrar el video en un Popup
        video = VideoPlayer(source=ruta_video, state='play')
        popup = Popup(title=archivo, content=video, size_hint=(None, None), size=(400, 400))
        popup.open()

class LoginScreen(Screen):
    pass


class HomeScreen(Screen):
    welcome_text = StringProperty('')


# Define la pantalla de registro
class RegisterScreen(Screen):
    pass


class LoginApp(App):

    def build(self):

        self.sm = ScreenManager()

        # Pantalla de inicio de sesión
        login_screen = LoginScreen(name='login')

        # iniciar_sesion_button = Button(text='Iniciar Sesión')
        # crear_cuenta_button = Button(text='Crear Cuenta')

        # iniciar_sesion_button.bind(on_press=self.iniciar_sesion)
        # crear_cuenta_button.bind(on_press=self.crear_cuenta)

        # layout_login = BoxLayout(orientation='vertical', spacing=10, padding=10)
        # layout_login.add_widget(iniciar_sesion_button)
        # layout_login.add_widget(crear_cuenta_button)

        # login_screen.add_widget(layout_login)

        # Pantalla principal (Home)
        home_screen = HomeScreen(name='home')

        # camara_button = Button(text='Cámara')
        # archivo_button = Button(text='Archivo')

        # camara_button.bind(on_press=self.abrir_camara)
        # archivo_button.bind(on_press=self.abrir_archivo)

        # layout_home = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # layout_home.add_widget(camara_button)
        # layout_home.add_widget(archivo_button)

        # home_screen.add_widget(layout_home)
        # Pantalla de registro
        register_screen = RegisterScreen(name='register')

        archivo_screen = ArchivoScreen(name='archivo')

        self.sm.add_widget(login_screen)
        self.sm.add_widget(home_screen)
        self.sm.add_widget(register_screen)
        self.sm.add_widget(archivo_screen)

        return self.sm

    def obtener_nombre(self):
        # Verificar si la carpeta 'fotos' ya existe
        carpeta_fotos = "fotos"
        if not os.path.exists(carpeta_fotos):
            # Si no existe, crear la carpeta
            os.makedirs(carpeta_fotos)

        # Con esto capturamos el nombre de la foto, de la persona que se haya logueado.
        # Ejemplo: 'Guillermo.jpg', extraemos solo el nombre "Guillermo"
        for nombre_archivo in os.listdir(carpeta_fotos):
            if nombre_archivo.endswith(".jpg"):
                nombre_sin_extension = os.path.splitext(nombre_archivo)[0]
                return nombre_sin_extension  # Devuelve el nombre sin extensión
    def iniciar_sesion(self):

        # Inicializar la cámara frontal
        cap = cv2.VideoCapture(0)

        # Variable para almacenar el nombre de la foto con coincidencia
        foto_coincidente = None

        # Obtener la lista de nombres de archivo de las fotos en la carpeta "foto"
        fotos_en_carpeta = os.listdir("fotos")

        while True:
            # Capturar un fotograma de la cámara frontal
            ret, frame = cap.read()

            # Mostrar el fotograma en una ventana
            cv2.imshow('Camara Frontal', frame)

            # Esperar a que el usuario presione la tecla 'q' para capturar la imagen
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Realizar el reconocimiento facial en la imagen capturada y comparar con las fotos en la carpeta "foto"
        try:
            for nombre_foto in fotos_en_carpeta:
                ruta_foto = os.path.join("fotos", nombre_foto)
                result = DeepFace.verify(ruta_foto, frame, model_name="Facenet", distance_metric='euclidean_l2')

                if result["verified"]:
                    print(f"La cara en la imagen capturada coincide con la foto {nombre_foto}.")
                    foto_coincidente = nombre_foto
                    break  # Detener la búsqueda cuando se encuentra una coincidencia

            if foto_coincidente:
                # Si hay una coincidencia, puedes guardar el nombre de la foto coincidente
                print("Foto coincidente:", foto_coincidente)
                self.sm.get_screen('home').welcome_text = self.obtener_nombre()
                # Cambia a la pantalla de inicio después de iniciar sesión
                self.sm.current = 'home'
            else:
                print("La cara en la imagen capturada no coincide con ninguna foto en la carpeta 'foto'.")

        except Exception as e:
            print("Error:", str(e))

        # Cerrar la ventana de la cámara
        cap.release()
        cv2.destroyAllWindows()

    def crear_cuenta(self):
        # Agregar lógica para crear una cuenta
        self.sm.current = 'register'

    def abrir_camara(self, instance):

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        posicion_izquierda = (50, 50)
        posicion_derecha = (50, 100)  # Nueva posición para el texto de "Grabando"

        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala_fuente = 1
        color = (255, 255, 255)

        # Directorio donde se guardarán las fotos y los videos
        output_directory = "fotos_y_videos/" + instance
        os.makedirs(output_directory, exist_ok=True)

        # Variables para la grabación de video
        grabando = False
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_output = None

        with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:

            start_time = None
            countdown = 5

            while True:
                ret, frame = cap.read()
                if ret == False:
                    break

                height, width, _ = frame.shape
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        if handedness.classification[0].label == "Left":
                            if start_time is None:
                                start_time = time.time()

                            elapsed_time = time.time() - start_time
                            remaining_time = countdown - elapsed_time

                            if remaining_time > 0:
                                texto = f"Mano izquierda - {int(remaining_time)}s"
                                cv2.putText(frame, texto, posicion_izquierda, fuente, escala_fuente, color, 2)
                            else:
                                filename = os.path.join(output_directory, f"foto_{int(time.time())}.jpg")
                                cv2.imwrite(filename, frame)
                                print(f"¡Foto tomada y guardada como {filename}!")
                                start_time = None

                        elif handedness.classification[0].label == "Right":
                            # Mostrar "Grabando" cuando se detecta la mano derecha
                            cv2.putText(frame, "Grabando", posicion_derecha, fuente, escala_fuente, color, 2)
                            if not grabando:
                                # Iniciar la grabación del video
                                filename = os.path.join(output_directory, f"video_{int(time.time())}.avi")
                                video_output = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
                                grabando = True
                        else:
                            if grabando:
                                # Detener la grabación del video
                                video_output.release()
                                grabando = False
                                print(f"Video guardado como {filename}")

                if grabando:
                    # Agregar el frame al video en grabación
                    video_output.write(frame)

                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Liberar recursos al finalizar
        if grabando:
            video_output.release()

        cap.release()
        cv2.destroyAllWindows()

    def abrir_archivo(self):
        # Obtener el resultado del método get_button_text()
        texto_boton = self.sm.get_screen('archivo').get_button_text()
        # Asignar el resultado al texto del botón
        self.sm.get_screen('archivo').ids.btn_archivo.text = texto_boton if texto_boton is not None else 'Vacío'
        # Cambiar a la pantalla de archivo
        self.sm.current = 'archivo'

    # Agrega una función para tomar la foto y guardarla con el nombre
    def tomar_foto(self, instance):
        nombre = instance  # Obtiene el nombre desde el TextInput
        if nombre:
            # Inicializar la cámara frontal
            cap = cv2.VideoCapture(0)

            while True:
                # Capturar un fotograma de la cámara frontal
                ret, frame = cap.read()

                # Mostrar el fotograma en una ventana
                cv2.imshow('Camara Frontal', frame)

                # Esperar a que el usuario presione la tecla 'q' para capturar la imagen
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Guardar la imagen con el nombre proporcionado en la carpeta "foto"
                    nombre_archivo = os.path.join("fotos", f"{nombre}.jpg")
                    cv2.imwrite(nombre_archivo, frame)

                    # Cerrar la ventana de la cámara
                    cap.release()
                    cv2.destroyAllWindows()

                    # Volver a la pantalla de inicio de sesión
                    self.sm.current = 'login'
                    break
        else:
            print("Ingresa un nombre válido antes de tomar la foto.")


if __name__ == '__main__':
    LoginApp().run()
