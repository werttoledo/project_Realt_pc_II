# Importaciones necesarias
import cv2
import mediapipe as mp # librería para el procesamiento de imágenes y videos
from utils import init_camera, draw_face_mesh # Funciones auxiliares


# Inicializar los modulos
mp_face_mesh = mp.solutions.face_mesh

# Configurar la camara
cap = init_camera() 

# Configurar el detector de rostros
with mp_face_mesh.FaceMesh(
    static_image_mode = False, # Procesamiento continuo -> para video
    max_num_faces = 2, # Numero maximo de rostros a detectar
    refine_landmarks = True, # Mayor detalle en ojos y labios
    # nivel de confianza
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:
    while True:
        ret, frame = cap.read() # leer un frame de la camara
        if not ret:
            print("No se pudo acceder a la camara")
            break

        # Convertir la imagen de BGR a RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la imagen para detectar los rostros
        results = face_mesh.process(rgb_frame)

        # Dibujar las mallas faciales en la imagen original
        frame = draw_face_mesh(frame, results)

        # Mostroar el resultado en pantalla 
        cv2.imshow("FaceMesh Detection", frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar la camara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()