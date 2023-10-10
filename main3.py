import os
from kivy.app import App
from kivy.properties import StringProperty, ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
import cv2
from deepface import DeepFace
import mediapipe as mp
import time
from kivy.uix.popup import Popup
from kivy.uix.image import Image
import easygui


class LoginScreen(Screen):
    pass


class HomeScreen(Screen):
    welcome_text = StringProperty('')


# Define la pantalla de registro
class RegisterScreen(Screen):
    pass


class FileScreen(Screen):
    my_list_attribute = ListProperty([])

    def on_my_list_attribute(self, instance, value):
        rv = self.ids.rv  # Obtén el RecycleView de la pantalla
        data = [{'archivo': archivo} for archivo in value]  # Crea los datos para el RecycleView
        rv.data = data  # Actualiza los datos del RecycleView


class BotonYArchivo(BoxLayout):
    archivo = StringProperty('')


class LoginApp(App):

    def build(self):

        self.sm = ScreenManager()

        # Pantalla de inicio de sesión
        login_screen = LoginScreen(name='login')

        # Pantalla principal (Home)
        home_screen = HomeScreen(name='home')

        # Pantalla de registro
        register_screen = RegisterScreen(name='register')

        file_screen = FileScreen(name='files')

        self.sm.add_widget(login_screen)
        self.sm.add_widget(home_screen)
        self.sm.add_widget(register_screen)
        self.sm.add_widget(file_screen)
        self.sm.current = 'login'

        return self.sm

    def iniciar_sesion(self):

        carpeta_fotos = "fotos"
        if not os.path.exists(carpeta_fotos):
            # Si no existe, crear la carpeta
            os.makedirs(carpeta_fotos)

        
        cap = cv2.VideoCapture(0)

        # Variable para almacenar el nombre de la foto con coincidencia
        foto_coincidente = None

        # Obtener la lista de nombres de archivo de las fotos en la carpeta "fotos"
        fotos_en_carpeta = os.listdir("fotos")

        while True:
            
            ret, frame = cap.read()

            
            cv2.imshow('Camara Frontal', frame)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Realizar el reconocimiento facial en la imagen capturada y comparar con las fotos en la carpeta "foto"
        try:
            for nombre_foto in fotos_en_carpeta:
                ruta_foto = os.path.join("fotos", nombre_foto)
                result = DeepFace.verify(ruta_foto, frame, model_name="Facenet", distance_metric='euclidean_l2')

                if result["verified"]:
                    easygui.msgbox(f"La cara en la imagen capturada coincide con la foto {nombre_foto}.",
                                   "Inicio de Sesión Exitoso")
                    # print(f"La cara en la imagen capturada coincide con la foto {nombre_foto}.")
                    foto_coincidente = nombre_foto
                    break  # Detener la búsqueda cuando se encuentra una coincidencia

            if foto_coincidente:
                
                print("Foto coincidente:", foto_coincidente)
                self.sm.get_screen('home').welcome_text = foto_coincidente
                
                self.sm.current = 'home'
            else:
                easygui.msgbox("La cara en la imagen capturada no coincide con ninguna foto en la carpeta 'foto'.",
                               "Inicio de Sesión Fallido")
                # print("La cara en la imagen capturada no coincide con ninguna foto en la carpeta 'foto'.")

        except Exception as e:
            error = str(e)
            easygui.msgbox(error, "Error")
            # print("Error:", str(e))

        
        cap.release()
        cv2.destroyAllWindows()

    def crear_cuenta(self):

        carpeta_fotos = "fotos"
        if not os.path.exists(carpeta_fotos):
            # Si no existe, crear la carpeta
            os.makedirs(carpeta_fotos)
        # Agregar lógica para crear una cuenta
        self.sm.current = 'register'

    def abrir_camara(self, instance):

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        # Variables de control
        posicion_izquierda = (50, 50)
        posicion_derecha = (50, 100)
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

                if grabando and results.multi_hand_landmarks is None:
                    # Detener la grabación del video

                    end_video_time = time.time()
                    video_duration = end_video_time - start_video_time
                    if video_duration >= 2.0:
                        video_output.release()
                        grabando = False
                        easygui.msgbox(f"Video guardado como {filename}", "mensaje")

                        # print(f"Video guardado como {filename}")
                    else:
                        # Borrar el video si dura menos de 2 segundos
                        video_output.release()
                        os.remove(filename)
                        grabando = False
                        # print(f"Video borrado porque dura menos de 2 segundos")

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
                                easygui.msgbox(f"¡Foto tomada y guardada como {filename}!", "mensaje")

                                # print(f"¡Foto tomada y guardada como {filename}!")
                                start_time = None

                        if handedness.classification[0].label == "Right":
                            cv2.putText(frame, "Grabando", posicion_derecha, fuente, escala_fuente, color, 2)
                            if not grabando:
                                # Iniciar la grabación del video
                                filename = os.path.join(output_directory, f"video_{int(time.time())}.avi")
                                video_output = cv2.VideoWriter(filename, fourcc, 20.0, (width, height))
                                grabando = True
                                start_video_time = time.time()
                            if grabando:
                                # Agregar el frame al video en grabación
                                video_output.write(frame)
                        # else:
                        if grabando and handedness.classification[0].label != "Right":
                            # Detener la grabación del video

                            end_video_time = time.time()
                            video_duration = end_video_time - start_video_time
                            if video_duration >= 2.0:
                                video_output.release()
                                grabando = False
                                easygui.msgbox(f"Video guardado como {filename}", "mensaje")

                                # print(f"Video guardado como {filename}")
                            else:
                                # Borrar el video si dura menos de 2 segundos
                                video_output.release()
                                os.remove(filename)
                                grabando = False
                                print(f"Video borrado porque dura menos de 2 segundos")

                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def listar_archivos(self, instance):
        carpeta = "fotos_y_videos/" + instance

        if not os.path.exists(carpeta):
            # Si no existe, crear la carpeta
            os.makedirs(carpeta)

        # Lista los archivos 
        archivos = os.listdir(carpeta)
        # Agrega el archivo a la lista de archivos
        self.sm.get_screen('files').my_list_attribute = archivos
        self.sm.current = 'files'

    # Agrega una función para tomar la foto y guardarla con el nombre
    def tomar_foto(self, instance):
        nombre = instance  # Obtiene el nombre desde el TextInput
        if nombre:
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                cv2.imshow('Camara Frontal', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # Guardar la imagen con el nombre proporcionado en la carpeta "fotos"
                    nombre_archivo = os.path.join("fotos", f"{nombre}.jpg")
                    cv2.imwrite(nombre_archivo, frame)
                    cap.release()
                    cv2.destroyAllWindows()

                    # Volver a la pantalla de inicio de sesión
                    self.sm.current = 'login'
                    break
        else:
            print("Ingresa un nombre válido antes de tomar la foto.")

    def abrir_fotos(self, archivo):
        home_screen = self.sm.get_screen('home')
        # Construye la ruta completa al archivo utilizando root.welcome_text
        carpeta = os.path.join("fotos_y_videos", home_screen.welcome_text)
        ruta_completa = os.path.join(carpeta, archivo)

        # Obtén la extensión del archivo
        extension = archivo.split('.')[-1].lower()

        if extension == 'jpg' or extension == 'jpeg' or extension == 'png':
            # Si es una foto, muestra la imagen
            self.mostrar_imagen(ruta_completa)
        elif extension == 'avi' or extension == 'mp4':
            # Si es un video, reproduce el video
            self.reproducir_video(ruta_completa)
        else:
            print("Tipo de archivo no compatible.")

    def mostrar_imagen(self, ruta_completa):
        try:
            # Carga y muestra la imagen en una ventana emergente
            imagen = Image(source=ruta_completa)
            popup = Popup(title='Imagen', content=imagen, size_hint=(None, None), size=(400, 400))
            popup.open()
        except Exception as e:
            print("Error al mostrar la imagen:", str(e))

    def reproducir_video(self, ruta_completa):
        # ruta completa esta guardada con este formato: fotos_y_videos\nombre.jpg\nombrevideo
        cadena = ruta_completa
        # le cambio las \ por /
        cadena_modificada = cadena.replace('\\', '/')

        try:
            captura = cv2.VideoCapture(cadena_modificada)
            while True:
                ret, imagen = captura.read()
                if ret:
                    cv2.imshow('video', imagen)
                else:
                    # El video ha llegado al final
                    break

                if cv2.waitKey(30) == 27:  # Presionar la tecla 'Esc' para salir
                    break
        except Exception as e:
            print(f"Ocurrió un error: {str(e)}")
        finally:
            captura.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    LoginApp().run()
