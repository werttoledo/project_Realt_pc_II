# Sistema de Reconocimiento Facial en Tiempo Real

Este proyecto implementa un sistema avanzado de reconocimiento facial utilizando tecnologías modernas como MediaPipe, OpenCV y SQLite para el procesamiento y almacenamiento de datos biométricos.

## 🌟 Flujo del Sistema

```
Captura ──→ Detección con MediaPipe
              │
              ↓
        Extracción de landmarks
              │
              ↓
        Comparación BD ←── Extracción de landmarks
              │
              ↓
     Resultado (nombre - desconocido)
              │
              ↓
         Mostrar  Almacenar
```

El sistema realiza las siguientes operaciones:
1. Captura de video en tiempo real
2. Detección facial usando MediaPipe
3. Extracción de landmarks (puntos faciales)
4. Comparación con la base de datos
5. Obtención del resultado (nombre de la persona o "desconocido")
6. Visualización del resultado y almacenamiento de nuevos rostros

## 📁 Estructura del Proyecto

```
project_Cpc_II/
│
├── src/
│   ├── main.py               # Cámara y análisis facial en tiempo real
│   ├── utils.py              # Funciones auxiliares
│   ├── db/
│   │   └── database.py       # Gestión de base de datos SQLite
│   └── recognition.py        # Módulo para comparación facial
│
├── data/
│   ├── captures/             # Imágenes de rostros registrados
│   └── embeddings/           # (opcional) Vectores faciales guardados
│
├── requirements.txt
└── README.md
```

## 🚀 Inicio Rápido

1. **Activación del Entorno Virtual**
   ```bash
   venv_mediapipe\Scripts\activate
   ```

2. **Ejecución del Programa**
   ```bash
   python src/main.py
   streamlit run app.py
   ```

## 🎮 Controles del Sistema

- **R**: Registrar un nuevo rostro
- **Q**: Salir del programa
- La detección y reconocimiento facial es automática y en tiempo real

## 💻 Componentes Principales

### 🎥 main.py
- Gestión de la cámara web
- Procesamiento de video en tiempo real
- Interfaz de usuario y visualización
- Control de flujo principal

### 🔧 utils.py
- Configuración de la cámara
- Funciones de dibujo para la visualización
- Utilidades de procesamiento de imagen

### 📊 database.py
- Gestión de la base de datos SQLite
- Almacenamiento de datos faciales
- Operaciones CRUD para rostros

## 🔍 recognition.py
- Algoritmos de comparación facial
- Procesamiento de landmarks faciales
- Lógica de reconocimiento
- **EMA (Exponential Moving Average)**: Suavizado dinámico de confianza en tiempo real
- **Normalización de landmarks**: Normalización min-max para reducir impacto de distancia/tamaño/movimiento

## 🛠️ Tecnologías Utilizadas

- **MediaPipe**: Framework de Google para detección facial
- **OpenCV**: Procesamiento de imágenes y video
- **SQLite**: Base de datos ligera y eficiente
- **NumPy**: Procesamiento numérico y arrays
- **Python**: Lenguaje de programación principal

## 📋 Requisitos del Sistema

- Python 3.10 o superior
- Cámara web funcional
- Espacio suficiente para la base de datos
- Memoria RAM recomendada: 4GB o superior

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del repositorio
2. Crea una nueva rama para tus cambios
3. Realiza tus modificaciones
4. Envía un pull request

## 📝 Notas Importantes

- La base de datos se crea automáticamente en la primera ejecución
- Se recomienda buena iluminación para mejor reconocimiento
- Los datos faciales se almacenan de forma segura localmente
- **Threshold bajo**: Si reconoce a personas que no son (falsos positivos), aumenta el threshold.
- **Threshold alto**: Si no reconoce a personas registradas (falsos negativos), reduce el threshold.
- **EMA bajo**: Si la barra de similitud fluctúa mucho, baja el alpha para suavizar.
- **EMA alto**: Si la barra es muy lenta, sube el alpha para más respuesta rápida.

## 🔒 Privacidad y Seguridad

- Los datos biométricos se almacenan localmente
- No se realizan conexiones a servicios externos
- Los datos son persistentes entre sesiones

## 📫 Contacto

Para dudas o sugerencias, puedes:
- Crear un issue en el repositorio
- Contactar al equipo de desarrollo

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.


# ./APP.PY
- sera el frontentod del proyecto
- Streamlit:
 - usa un flujo secuencial (de arriba hacia abajo)
 - reaccion en tiempo real a widgets (button, selectbox, slider)



 #secciones
 - streamlit.sidebar.radio() => Menu lateral con las tres vistas principales
- reconocimiento en vivo => abre camara => usar medipipe => muestra nombre => barra de similitud
- Registro de nuevo rostro => permite ingresar nombre => toma la foto => extraer los landmarks
 - Ver registros -> lista nombres e imagenes guardadas -> data/captures/{nameImage}





