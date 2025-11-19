# Bibliotecas estándar
import warnings; warnings.filterwarnings("ignore")
import cv2, mediapipe as mp
import numpy as np
import os
from utils import init_camera, draw_face_mesh, draw_similarity_bar # Funciones auxiliares
from db.database import init_db, insert_face
from recognition import compare_faces



mp_face_mesh = mp.solutions.face_mesh # Inicializar los modulos


#Parametro de calibracion
THRESHOLD = 0.6 # umbral de similitud para el match
ALPHA_SMOOTHING = 0.25 # factor de suavizado para la barra de similitud


#Definir la funcion main principal 
def main():

    # Inicializar la base de datos
    init_db()
    
    # Crear carpeta para captures si no existe
    os.makedirs("data/captures", exist_ok=True)
    
    cap = init_camera() # Configurar la camara

    #Estado para suavisando
    smooth_confidence = 0.0
    last_name = "Desconocido"

    # Configurar el detector de rostros
    with mp_face_mesh.FaceMesh(
        static_image_mode = False, # Procesamiento continuo -> para video
        max_num_faces = 3, # Numero maximo de rostros a detectar
        refine_landmarks = True, # Mayor detalle en ojos y labios
        min_detection_confidence = 0.5, # nivel de confianza
        min_tracking_confidence = 0.5
    ) as face_mesh:
        while True:
            ok, frame = cap.read() # leer un frame de la camara
            if not ok:
                print("No se pudo acceder a la camara")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir la imagen de BGR a RGB
            results = face_mesh.process(rgb) # Procesar la imagen para detectar los rostros
            
            frame = draw_face_mesh(frame, results)
            curr_conf = 0.0
            curr_name = "Desconocido"

            # Extraccion de los landmarks faciales
            if results.multi_face_landmarks:
                # solo usamos el primer rostro para la brra global:
                #igualmente el puede dibujar nombre y barra del rostro 

                first = results.multi_face_landmarks[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in first.landmark]

                curr_name, dist = compare_faces(landmarks, threshold=THRESHOLD)
                #Confianza simple inversa: 1 - dist (acotada a [0,1])
                curr_conf = max(0.0, min(1.0, 1.0 - dist))

                # Suavizado exponencial de la confianza y el nombre
                smooth_confidence = (ALPHA_SMOOTHING * curr_conf +
                                     (1 - ALPHA_SMOOTHING) * smooth_confidence)
                
                if curr_name != "Desconocido":
                    last_name = curr_name

                #overlay de texto 
                cv2.putText(frame, f"{last_name}", (30, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, "Confianza", (30, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                #barra de similitud -> verde
                frame = draw_similarity_bar(frame, smooth_confidence, x=30, y=110, w=260, h=18)

                #ventana
                cv2.imshow("Reconocimiento Facial (r=registar, q=salir)", frame)


 # Captura de teclas
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # Si se presiona 'r' -> registrar nuevo rostro
            if key == ord('r'):
                if results.multi_face_landmarks:
                    first = results.multi_face_landmarks[0]
                    landmarks = [(lm.x, lm.y, lm.z) for lm in first.landmark]
                    name = input("Nombre de la persona: ")

                    if name:
                        insert_face(name, landmarks)
                        cv2.imwrite(f"data/captures/{name}.png", frame)
                        print(f"✅ Rostro registrado para {name}")
                        last_name = name
                        smooth_confidence = 1.0  # Confianza máxima tras registro
                    else:
                        print("❌ Nombre vacío, no se registró el rostro.")
                else:
                    print("❌ No se detectó ningún rostro para registrar.")

    # Liberar la cámara y cerrar las ventanas
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
- Distancia Euclidiana: Cuanto más pequeña, más parecido
- Umbral (THRESHOLD): si la mejor distancia < umbral -> "reconocido"
- Confianza (EMA): ALPHA_SMOOTH evita que la barra "salte" por ruido
- Barra: representación visual inmediata para el usuario final
"""
