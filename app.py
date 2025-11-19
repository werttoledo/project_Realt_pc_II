import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import sys

# Ensure 'src' directory is on sys.path so imports like `from db.database` work
BASE_DIR = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASE_DIR, "src")
if os.path.isdir(SRC_DIR) and SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from db.database import init_db, insert_face, get_all_faces, delete_face_by_id, update_face_by_id, update_name_by_id
from recognition import compare_faces
from utils import (
    draw_similarity_bar,
    init_camera,
    apply_illumination_filter,
    auto_zoom_on_face,
    remap_landmarks_to_bbox,
)
import recognition as recognition_module
from datetime import datetime

# Inicializar base de datos y carpetas
init_db()
os.makedirs("data/captures", exist_ok=True)

# Configuración básica de la app
st.set_page_config(page_title="Reconocimiento Facial con MediaPipe", layout="wide")
st.markdown("Esta aplicación muestra el uso de **MediaPipe** para reconocer rostros en tiempo real usando la cámara")

# Estilos básicos (tema ligero y responsive mínimo)
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1724; color: #e6eef8; }
    .css-1d391kg { color: #e6eef8; }
    .stButton>button { background-color: #0ea5a4; color: white; }
    .stSlider>div>div>div>input { accent-color: #0ea5a4; }
    .card { background: linear-gradient(135deg,#0b1220 0%, #11233a 100%); padding: 12px; border-radius:10px }
    @media (max-width: 600px) {
        .card { padding: 8px }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Menú lateral
menu = st.sidebar.radio("Selecciona una opción:", ["Home", "Reconocimiento en vivo", "Registrar nuevo rostro", "Ver registro"], key="main_menu")

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# --- Home: tarjeta de usuario reciente
if menu == "Home":
    st.title("Home")
    st.write("Resumen rápido del sistema y usuario más reciente registrados")

    registros = get_all_faces()
    latest = None
    if registros:
        # Preferir created_at si está presente (filas con 4 elementos)
        try:
            # Filas posibles: (id,name,landmarks,created_at) o (id,name,landmarks)
            with_dates = [r for r in registros if len(r) >= 4]
            if with_dates:
                # ordenar por created_at (ISO compatible)
                latest = sorted(with_dates, key=lambda r: r[3] or "", reverse=True)[0]
            else:
                latest = registros[-1]
        except Exception:
            latest = registros[-1]

    if latest:
        if len(latest) >= 4:
            row_id, name, _, created_at = latest
        else:
            row_id, name, _ = latest
            created_at = None

        col_img, col_info = st.columns([1, 3])
        img_path = f"data/captures/{name}.jpg"
        with col_img:
            if os.path.exists(img_path):
                st.image(img_path, width=180)
            else:
                st.caption("Sin imagen")
        with col_info:
            st.markdown(f"### {name}")
            if created_at:
                st.write(f"**Registrado:** {created_at}")
            else:
                st.write("**Registrado:** Fecha no disponible")
            st.write("Estado: **Activo**")
    else:
        st.info("Aún no hay usuarios registrados. Ve a 'Registrar nuevo rostro' para añadir uno.")


#############################
# OPCIÓN 1: RECONOCIMIENTO EN VIVO
#############################
if menu == "Reconocimiento en vivo":
    st.subheader("Cámara en tiempo real")

    # Controles rápidos
    cols_ctrl = st.columns([1, 1, 1, 1])
    with cols_ctrl[0]:
        run = st.checkbox("Iniciar cámara", key="run_checkbox")
    with cols_ctrl[1]:
        threshold = st.slider("Umbral (threshold)", 0.0, 1.0, 0.6, 0.01, key="threshold_slider")
    with cols_ctrl[2]:
        alpha_val = st.slider("EMA alpha", 0.0, 1.0, 0.6, 0.01, key="alpha_slider")
        # actualizar alpha global en el módulo recognition
        try:
            recognition_module.alpha = float(alpha_val)
        except Exception:
            pass
    with cols_ctrl[3]:
        long_range = st.checkbox("Modo Largo Alcance", key="long_range")

    FRAME_WINDOW = st.image([], clamp=True)

    # Opciones adicionales
    enhance_illum = st.checkbox("Mejorar iluminación/contraste", key="enhance_illum")
    auto_zoom = st.checkbox("Auto-zoom sobre rostro", key="auto_zoom")
    refresh_camera = st.button("Detener / Refrescar cámara", key="refresh_cam")

    # Determinar resolución según modo
    if long_range:
        cam_w, cam_h = 1280, 720
    else:
        cam_w, cam_h = 640, 480

    # Reiniciar flag si se pide refresh: usamos un flag alternativo para evitar modificar directamente
    # el widget `run_checkbox` después de su creación (Streamlit lo prohíbe).
    if refresh_camera:
        st.session_state["stop_camera"] = True

    cap = init_camera(cam_w, cam_h)

    try:
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("No se puede acceder a la cámara")
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                # Procesar detección
                if results and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extraer landmarks
                        landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]

                        name, confidence = compare_faces(landmarks, threshold=threshold)

                        # Aplicar mejora de iluminación si se indicó
                        display_frame = frame
                        if enhance_illum:
                            display_frame = apply_illumination_filter(display_frame, method='clahe')

                        landmarks_to_draw = face_landmarks

                        # Auto-zoom: recortar y reescalar antes de dibujar
                        if auto_zoom:
                            base_frame_for_zoom = display_frame
                            zoomed, bbox = auto_zoom_on_face(
                                base_frame_for_zoom,
                                landmarks,
                                output_size=(cam_w, cam_h),
                                padding=0.4,
                                return_metadata=True,
                            )
                            display_frame = zoomed
                            if bbox:
                                landmarks_to_draw = remap_landmarks_to_bbox(
                                    face_landmarks, bbox, base_frame_for_zoom.shape
                                )

                        # Dibujar malla (si no estamos mostrando zoomed, dibujar sobre frame)
                        try:
                            mp_drawing.draw_landmarks(
                                display_frame,
                                landmarks_to_draw,
                                mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
                            )
                        except Exception:
                            pass

                        # Barra de similitud
                        display_frame = draw_similarity_bar(display_frame, confidence, x=30, y=30, w=200, h=14)

                        # Nombre
                        cv2.putText(display_frame, f"{name}", (30, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # Mostrar
                        FRAME_WINDOW.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                        continue

                # Si no hubo rostros detectados, mostrar frame normal (posible filtro aplicado)
                if not (results and results.multi_face_landmarks):
                    frame_to_show = frame
                    if enhance_illum:
                        frame_to_show = apply_illumination_filter(frame_to_show, method='clahe')
                    FRAME_WINDOW.image(cv2.cvtColor(frame_to_show, cv2.COLOR_BGR2RGB))

                # Check stop flag (set when user clicks Detener/Refrescar)
                if st.session_state.get("stop_camera", False):
                    # Clear the flag and break gracefully
                    st.session_state["stop_camera"] = False
                    break

                if not st.session_state.get("run_checkbox", True):
                    break
    finally:
        if cap:
            cap.release()

#############################
# OPCIÓN 2: REGISTRAR NUEVO ROSTRO
#############################
elif menu == "Registrar nuevo rostro":
    st.subheader("Registrar un nuevo rostro")

    name = st.text_input("Nombre del usuario", key="name_input")
    capture = st.button("Capturar rostro", key="capture_btn")

    if capture and name:
        cap = init_camera(1280, 720)
        st.info("Capturando imagen...")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("No se pudo capturar desde la cámara")
        else:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as fm:
                results = fm.process(rgb)

                if results and results.multi_face_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark]

                    insert_face(name, landmarks)
                    cv2.imwrite(f"data/captures/{name}.jpg", frame)

                    st.success(f"Rostro guardado correctamente como {name}")
                else:
                    st.warning("No se detectó ningún rostro. Intenta nuevamente.")

#############################
# OPCIÓN 3: VER REGISTRO
#############################
elif menu == "Ver registro":
    st.subheader("Rostros registrados en la base de datos")

    registros = get_all_faces()

    if not registros:
        st.warning("No hay registros todavía")
    else:
        for row in registros:
            # row may be (id, name, landmarks, created_at) or (id, name, landmarks)
            if len(row) >= 4:
                row_id, name, _, created_at = row
            else:
                row_id, name, _ = row
                created_at = None

            col1, col2, col3 = st.columns([1, 2, 2])

            img_path = f"data/captures/{name}.jpg"

            # IMAGEN
            with col1:
                if os.path.exists(img_path):
                    st.image(img_path, width=120)
                else:
                    st.caption("Sin imagen")

            # INFO + ELIMINAR
            with col2:
                st.write(f"**{name}**")
                if created_at:
                    st.write(f"_Registrado:_ {created_at}")

                # Keys built with row_id to ensure uniqueness even if names repeat
                delete_btn_key = f"del_{row_id}"
                confirm_flag_key = f"to_delete_{row_id}"
                confirm_btn_key = f"conf_{row_id}"

                if st.button(f"Eliminar {name}", key=delete_btn_key):
                    # set a session flag to show inline confirmation
                    st.session_state[confirm_flag_key] = True

                if st.session_state.get(confirm_flag_key, False):
                    st.warning(f"¿Seguro que deseas eliminar a {name}?")
                    if st.button(f"Confirmar eliminar {name}", key=confirm_btn_key):
                        # delete by id to avoid removing multiple entries with same name
                        delete_face_by_id(row_id)
                        if os.path.exists(img_path):
                            os.remove(img_path)
                        st.success(f"{name} eliminado correctamente")
                        # clear confirmation flag
                        st.session_state[confirm_flag_key] = False

            # ACTUALIZAR
            with col3:
                update_btn_key = f"upd_{row_id}"
                update_flag_key = f"to_update_{row_id}"
                capture_update_key = f"capture_update_{row_id}"
                save_update_key = f"save_update_{row_id}"
                edit_name_key = f"edit_name_{row_id}"
                preview_tmp_key = f"tmp_image_{row_id}"
                tmp_landmarks_key = f"tmp_landmarks_{row_id}"

                if st.button(f"Actualizar {name}", key=update_btn_key):
                    st.session_state[update_flag_key] = True

                if st.session_state.get(update_flag_key, False):
                    st.info("Editar registro")
                    # Nombre editable
                    new_name = st.text_input("Nuevo nombre", value=name, key=edit_name_key)

                    # Botón para capturar nueva imagen
                    if st.button("Capturar nueva imagen", key=capture_update_key):
                        st.info("Capturando nueva imagen...")
                        cap = init_camera(640, 480)
                        ret, frame = cap.read()
                        cap.release()

                        if not ret:
                            st.error("No se pudo acceder a la cámara")
                        else:
                            frame = cv2.flip(frame, 1)
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                            with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as fm:
                                results = fm.process(rgb)

                                if results and results.multi_face_landmarks:
                                    landmarks = [(lm.x, lm.y, lm.z)
                                                 for lm in results.multi_face_landmarks[0].landmark]

                                    tmp_path = f"data/captures/{row_id}_tmp.jpg"
                                    cv2.imwrite(tmp_path, frame)
                                    st.session_state[tmp_landmarks_key] = landmarks
                                    st.session_state[preview_tmp_key] = tmp_path
                                    st.success("Imagen capturada. Previsualización abajo.")
                                else:
                                    st.error("No se detectó ningún rostro. Intenta de nuevo.")

                    # Previsualizar si hay captura temporal
                    if st.session_state.get(preview_tmp_key):
                        st.image(st.session_state[preview_tmp_key], width=160)

                    # Guardar cambios
                    if st.button("Guardar cambios", key=save_update_key):
                        # Obtener valores
                        new_name_val = st.session_state.get(edit_name_key, name)
                        new_landmarks = st.session_state.get(tmp_landmarks_key)

                        # Si el nombre cambió, actualizar y renombrar la imagen si existe
                        if new_name_val and new_name_val != name:
                            old_path = f"data/captures/{name}.jpg"
                            new_path = f"data/captures/{new_name_val}.jpg"
                            if os.path.exists(old_path):
                                try:
                                    os.replace(old_path, new_path)
                                except Exception:
                                    try:
                                        os.remove(old_path)
                                    except Exception:
                                        pass
                            update_name_by_id(row_id, new_name_val)
                            name = new_name_val

                        # Si hay nuevos landmarks, actualizar y mover la imagen temporal
                        if new_landmarks:
                            update_face_by_id(row_id, new_landmarks)
                            final_path = f"data/captures/{new_name_val or name}.jpg"
                            tmp_path = st.session_state.get(preview_tmp_key)
                            if tmp_path and os.path.exists(tmp_path):
                                try:
                                    os.replace(tmp_path, final_path)
                                except Exception:
                                    try:
                                        os.remove(tmp_path)
                                    except Exception:
                                        pass

                        st.success("Registro actualizado correctamente")
                        # Limpiar estado temporal
                        st.session_state[update_flag_key] = False
                        for k in (tmp_landmarks_key, preview_tmp_key):
                            if k in st.session_state:
                                del st.session_state[k]
