import numpy as np
from db.database import get_all_faces
from utils import normalize_landmarks

# Valores globales para EMA
ema_confidence = None
alpha = 0.6  # controla el suavizado (0 suave – 1 muy rápido)


def compare_faces(new_landmarks, threshold=0.6):
    """
    Compara los landmarks nuevos con los registros almacenados.

    Retorna:
        (nombre_detectado, confianza_suavizada_ema)
    """

    global ema_confidence

    faces = get_all_faces()
    if not faces:
        return "Sin registros", 0.0

    # Normalizar nuevos landmarks
    new_vec = normalize_landmarks(new_landmarks)

    best_name = "Desconocido"
    best_similarity = 0.0

    # get_all_faces may return rows as (id, name, landmarks) or (id, name, landmarks, created_at)
    for row in faces:
        # Support multiple row shapes defensively
        if isinstance(row, (list, tuple)):
            if len(row) >= 3:
                # id, name, landmarks, (created_at)
                _, name, landmarks_str = row[0], row[1], row[2]
            elif len(row) == 2:
                # legacy shape: name, landmarks
                name, landmarks_str = row
            else:
                # unexpected shape, skip
                continue
        else:
            # not iterable, skip
            continue
        try:
            db_list = eval(landmarks_str)  # convertir string a lista
            db_vec = normalize_landmarks(db_list)

            # Igualar longitudes
            min_len = min(len(new_vec), len(db_vec))
            new_vec_cmp = new_vec[:min_len]
            db_vec_cmp = db_vec[:min_len]

            # Distancia
            dist = np.linalg.norm(new_vec_cmp - db_vec_cmp)

            # Convertir distancia en similitud (0 a 1)
            similarity = 1.0 / (1.0 + float(dist))

            if similarity > best_similarity:
                best_similarity = similarity
                best_name = name

        except Exception as e:
            print(f"[ERROR] comparando con {name}: {e}")
            continue

    # Aplicar suavizado EMA
    if ema_confidence is None:
        ema_confidence = best_similarity
    else:
        ema_confidence = alpha * best_similarity + (1 - alpha) * ema_confidence

    # Validar threshold
    if best_similarity >= threshold:
        return best_name, float(ema_confidence)
    else:
        return "Desconocido", float(ema_confidence)
