import sqlite3
import os

DB_PATH = "data/face_data.db"

def init_db():
    """Crear la base de datos y la tabla si no existen """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Crear tabla con columna created_at (timestamp) si no existe
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS faces(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            landmarks TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )'''
    )

    # Si la tabla existe sin la columna created_at, agregarla (compatibilidad)
    cursor.execute("PRAGMA table_info(faces)")
    cols = [r[1] for r in cursor.fetchall()]
    if 'created_at' not in cols:
        try:
            cursor.execute("ALTER TABLE faces ADD COLUMN created_at TEXT DEFAULT (datetime('now'))")
        except Exception:
            # Si no es posible alterar (versión antigua), ignorar
            pass
    conn.commit()
    conn.close()

def insert_face(name, landmarks):
    """Insertar un nuevo rostro (nombre, landmarks) en la tabla faces"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # created_at usa DEFAULT CURRENT_TIMESTAMP a nivel de SQLite
    cursor.execute(
        "INSERT INTO faces (name, landmarks) VALUES (?, ?)",
        (name, str(landmarks)))
    conn.commit()
    conn.close()

def get_all_faces():
    """Obtener todos los rostros de la tabla faces"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Intentar devolver id, name, landmarks y created_at si existe
    try:
        cursor.execute("SELECT id, name, landmarks, created_at FROM faces")
        faces = cursor.fetchall()
    except Exception:
        # Tabla antigua sin created_at
        cursor.execute("SELECT id, name, landmarks FROM faces")
        faces = cursor.fetchall()
    conn.close()
    return faces

def delete_face(name):
    """Eliminar un rostro por su nombre."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faces WHERE name = ?", (name,))
    conn.commit()
    conn.close()

def delete_face_by_id(row_id):
    """Eliminar un rostro por su id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM faces WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()

def update_face(name, new_landmarks):
    """Actualizar landmarks de un rostro ya registrado."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE faces SET landmarks = ? WHERE name = ?",
        (str(new_landmarks), name)
    )
    conn.commit()
    conn.close()

def update_face_by_id(row_id, new_landmarks):
    """Actualizar landmarks de un rostro por id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE faces SET landmarks = ? WHERE id = ?",
        (str(new_landmarks), row_id)
    )
    conn.commit()
    conn.close()

def update_name_by_id(row_id, new_name):
    """Actualizar el nombre de un rostro por su id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE faces SET name = ? WHERE id = ?",
        (new_name, row_id)
    )
    conn.commit()
    conn.close()
