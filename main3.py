import os
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
from kivy.uix.image import Image
from kivy.uix.videoplayer import VideoPlayer



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


        self.sm.add_widget(login_screen)
        self.sm.add_widget(home_screen)
        self.sm.add_widget(register_screen)

        return self.sm

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
                self.sm.get_screen('home').welcome_text = foto_coincidente
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
        # Agregar lógica para abrir la funcionalidad de archivo
        pass

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
