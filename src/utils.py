import cv2
import mediapipe as mp


# Inicializar los modulos de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def init_camera(width=640, height=480):
    """Inicializa la cámara con dimensiones personalizadas."""
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    return cap

def draw_face_mesh(frame, results):
    """Dibuja la malla facial (color verde) sobre el frame."""
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )
    return frame