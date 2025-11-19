import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

# Inicializar módulos de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def init_camera(width=640, height=480, device=0):
    """Inicializa la cámara con dimensiones personalizadas."""
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def draw_face_mesh(frame, results):
    """Dibuja la malla facial (color verde) sobre el frame."""
    if results and results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
            )
    return frame


def draw_similarity_bar(frame, confidence, x=30, y=30, w=220, h=18):
    """Dibuja barra de similitud en la pantalla."""
    confidence = float(max(0.0, min(1.0, confidence)))

    # Marco
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Relleno verde
    fill = int(w * confidence)
    cv2.rectangle(frame, (x + 2, y + 2), (x + 2 + fill, y + h - 2), (0, 255, 0), -1)

    # Texto
    cv2.putText(
        frame,
        f"{confidence * 100:.1f}%",
        (x + w + 10, y + h - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    return frame


def normalize_landmarks(landmarks):
    """
    Normaliza los landmarks para reducir el impacto de distancia, tamaño o movimiento.
    Devuelve un vector 1D listo para comparación.
    """
    lm = np.array(landmarks, dtype=float)

    if lm.ndim != 2 or lm.shape[1] != 3:
        return lm.flatten()

    x, y, z = lm[:, 0], lm[:, 1], lm[:, 2]

    # Normalización min-max
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)

    lm_norm = np.stack([x_norm, y_norm, z_norm], axis=1)
    return lm_norm.flatten()


def apply_illumination_filter(frame, method='clahe', clip_limit=3.0, tile_grid_size=(8,8), gamma=None):
    """
    Mejora la iluminación/contraste del frame.
    - method 'clahe': aplica CLAHE sobre el canal L (LAB)
    - gamma: si se especifica, aplica corrección gamma antes de CLAHE
    """
    img = frame.copy()

    if gamma is not None and gamma > 0:
        invGamma = 1.0 / float(gamma)
        table = (np.array([((i / 255.0) ** invGamma) * 255
                           for i in np.arange(0, 256)])).astype("uint8")
        img = cv2.LUT(img, table)

    if method == 'clahe':
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        res = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return res

    return img


def auto_zoom_on_face(frame, landmarks, output_size=(640, 480), padding=0.4, return_metadata=False):
    """
    Recorta y hace zoom automáticamente alrededor del rostro detectado basado en landmarks.
    - landmarks: lista de (x,y,z) normalizados (0..1)
    - output_size: tamaño final del frame devuelto
    """
    h, w = frame.shape[:2]
    lm = np.array(landmarks)
    if lm.size == 0:
        return (frame, None) if return_metadata else frame

    # Extraer solo x,y
    xy = lm[:, :2]
    xs = xy[:, 0] * w
    ys = xy[:, 1] * h

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())

    # Expandir bbox con padding
    bw = max(1, max_x - min_x)
    bh = max(1, max_y - min_y)
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    x1 = max(0, min_x - pad_x)
    y1 = max(0, min_y - pad_y)
    x2 = min(w - 1, max_x + pad_x)
    y2 = min(h - 1, max_y + pad_y)

    if x2 <= x1 or y2 <= y1:
        return (frame, None) if return_metadata else frame

    crop = frame[y1:y2, x1:x2]
    try:
        resized = cv2.resize(crop, (output_size[0], output_size[1]))
        if return_metadata:
            return resized, (x1, y1, x2, y2)
        return resized
    except Exception:
        return (frame, None) if return_metadata else frame


def remap_landmarks_to_bbox(face_landmarks, bbox, frame_shape):
    """
    Recalcula los landmarks normalizados para que se alineen con un recorte específico.
    - face_landmarks: objeto NormalizedLandmarkList
    - bbox: (x1, y1, x2, y2) en píxeles relativos al frame original
    - frame_shape: alto y ancho del frame original (antes del recorte)
    """
    if bbox is None or face_landmarks is None:
        return face_landmarks

    x1, y1, x2, y2 = bbox

    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)
    h, w = frame_shape[:2]

    remapped = landmark_pb2.NormalizedLandmarkList()

    for lm in face_landmarks.landmark:
        abs_x = lm.x * w
        abs_y = lm.y * h

        rel_x = (abs_x - x1) / crop_w
        rel_y = (abs_y - y1) / crop_h

        rel_x = float(min(max(rel_x, 0.0), 1.0))
        rel_y = float(min(max(rel_y, 0.0), 1.0))

        remapped.landmark.append(
            landmark_pb2.NormalizedLandmark(
                x=rel_x,
                y=rel_y,
                z=lm.z,
                visibility=lm.visibility,
                presence=lm.presence,
            )
        )

    return remapped
