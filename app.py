import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import os
import json
import joblib
from PIL import Image

# Konstanta (diterjemahkan ke dalam bahasa Indonesia)
UPLOAD_IMAGE_TEXT = "Unggah gambar"
FACE_DETECTION_TEXT = "Deteksi Wajah"
COMPRESSION_COMPARISON_TEXT = "Perbandingan Kompresi"
COLOR_SPACE_CONVERSIONS_TEXT = "Konversi Ruang Warna"
TEXTURE_ANALYSIS_TEXT = "Analisis Tekstur"
GESTURE_CONTROL_DRONE_TEXT = "Kontrol Gerakan & Simulasi Drone"

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Pengaturan direktori
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

GESTURE_DIR = "gestures"
if not os.path.exists(GESTURE_DIR):
    os.makedirs(GESTURE_DIR)

GESTURE_JSON = os.path.join(GESTURE_DIR, "gestures.json")
if os.path.exists(GESTURE_JSON):
    with open(GESTURE_JSON, 'r') as f:
        gesture_db = json.load(f)
else:
    gesture_db = []

# Memuat model pengenalan gerakan yang telah dilatih sebelumnya
try:
    model = joblib.load('gesture_classifier_cv.pkl')
except FileNotFoundError:
    st.error("Model terlatih 'gesture_classifier_cv.pkl' tidak ditemukan. Silakan latih model terlebih dahulu.")
    model = None

# Kelas Simulator Drone (diterjemahkan ke dalam bahasa Indonesia)
class DroneSimulator:
    GESTURE_TAKEOFF, GESTURE_LAND, GESTURE_FORWARD, GESTURE_BACKWARD, GESTURE_LEFT, GESTURE_RIGHT, GESTURE_UP, GESTURE_DOWN, GESTURE_ROTATE_CW, GESTURE_ROTATE_CCW, GESTURE_NONE = range(0, 11)
    MOVE_SPEED = 0.2
    ROTATE_SPEED = 0.2

    def __init__(self):
        self.position = {"x": 0, "y": 0, "z": 0}
        self.rotation = {"x": 0, "y": 0, "z": 0}
        self.is_flying = False
        self.last_gesture = self.GESTURE_NONE
        self.gesture_labels = {i: name for i, name in enumerate(["TERBANG", "MENDARAT", "MAJU", "MUNDUR", "KIRI", "KANAN", "NAIK", "TURUN", "PUTAR KANAN", "PUTAR KIRI", "TIDAK ADA"])}

    def update_state_from_gesture(self, gesture_id):
        if gesture_id == self.GESTURE_TAKEOFF and not self.is_flying:
            self.is_flying = True
        elif gesture_id == self.GESTURE_LAND and self.is_flying:
            self.is_flying = False
            self.position["y"] = 0
        if self.is_flying:
            if gesture_id == self.GESTURE_FORWARD:
                self.position["x"] += self.MOVE_SPEED * np.cos(self.rotation["y"])
                self.position["z"] -= self.MOVE_SPEED * np.sin(self.rotation["y"])
            elif gesture_id == self.GESTURE_BACKWARD:
                self.position["x"] -= self.MOVE_SPEED * np.cos(self.rotation["y"])
                self.position["z"] += self.MOVE_SPEED * np.sin(self.rotation["y"])
            elif gesture_id == self.GESTURE_LEFT:
                self.position["x"] -= self.MOVE_SPEED * np.sin(self.rotation["y"])
                self.position["z"] -= self.MOVE_SPEED * np.cos(self.rotation["y"])
            elif gesture_id == self.GESTURE_RIGHT:
                self.position["x"] += self.MOVE_SPEED * np.sin(self.rotation["y"])
                self.position["z"] += self.MOVE_SPEED * np.cos(self.rotation["y"])
            elif gesture_id == self.GESTURE_UP:
                self.position["y"] += self.MOVE_SPEED
            elif gesture_id == self.GESTURE_DOWN:
                self.position["y"] -= self.MOVE_SPEED
            elif gesture_id == self.GESTURE_ROTATE_CW:
                self.rotation["y"] += self.ROTATE_SPEED
            elif gesture_id == self.GESTURE_ROTATE_CCW:
                self.rotation["y"] -= self.ROTATE_SPEED
        self.last_gesture = gesture_id
        return self.get_state()

    def get_state(self):
        return {
            "posisi": self.position,
            "rotasi": self.rotation,
            "terbang": self.is_flying,
            "gerakan": self.gesture_labels.get(self.last_gesture, "TIDAK DIKENAL")
        }

# Fungsi Pembantu
def display_before_after(original, processed, title):
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Asli", use_column_width=True)
    with col2:
        st.image(processed, caption=title, use_column_width=True)

# CSS Khusus untuk Peningkatan UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }
    .stApp {
        background-color: #1e3a5f;
        color: #f0f0f0;
    }
    .stButton>button {
        background-color: #f0a500;
        color: #1e3a5f;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 700;
    }
    .stSelectbox {
        background-color: #f0a500;
        color: #1e3a5f;
        border-radius: 5px;
    }
    .stSlider {
        color: #f0a500;
    }
    .stImage {
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #f0a500;
    }
    </style>
""", unsafe_allow_html=True)

# Fungsi Halaman
def home_page():
    st.title("Selamat Datang di Aplikasi Pemrosesan Gambar")
    st.write("Aplikasi ini mengintegrasikan berbagai fitur pemrosesan gambar, penglihatan komputer, dan pengenalan gerakan. Gunakan bilah samping untuk menavigasi fitur-fiturnya.")
    
    # Unggah Gambar di Halaman Beranda
    uploaded_file = st.file_uploader(UPLOAD_IMAGE_TEXT, type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.session_state['uploaded_image'] = image
        st.image(image, caption="Gambar yang Diunggah", use_column_width=True)
    else:
        st.write("Silakan unggah gambar untuk digunakan di fitur lain.")

def image_processing_page():
    st.title("Pemrosesan Gambar Dasar")
    if 'uploaded_image' not in st.session_state:
        st.write("Tidak ada gambar yang diunggah. Silakan unggah gambar di halaman Beranda.")
        return
    image = st.session_state['uploaded_image']
    
    processing_type = st.selectbox("Pilih Pemrosesan", ["Tidak Ada", "Grayscale", "Negatif", "Histogram"])
    
    if processing_type != "Tidak Ada":
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Gambar Asli", use_column_width=True)
        
        if processing_type == "Grayscale":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            with col2:
                st.image(gray, caption="Gambar Grayscale", use_column_width=True)
        elif processing_type == "Negatif":
            negative = 255 - image
            with col2:
                st.image(negative, caption="Gambar Negatif", use_column_width=True)
        elif processing_type == "Histogram":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fig, ax = plt.subplots()
            ax.hist(gray.ravel(), 256, [0, 256])
            with col2:
                st.pyplot(fig)

def apply_convolution(image):
    kernel_type = st.selectbox("Tipe Kernel", ["rata-rata", "tajam", "tepi"])
    if st.button("Terapkan Konvolusi"):
        kernels = {
            "rata-rata": np.ones((3, 3), np.float32) / 9,
            "tajam": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            "tepi": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        }
        result = cv2.filter2D(image, -1, kernels[kernel_type])
        display_before_after(image, result, f"Konvolusi {kernel_type.capitalize()}")

def apply_padding(image):
    padding_size = st.slider("Ukuran Padding", 1, 50, 10)
    padding_type = st.selectbox("Tipe Padding", ["Padding Nol", "Replikasi", "Refleksi"])
    if st.button("Terapkan Padding"):
        if padding_type == "Padding Nol":
            result = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT, value=0)
        elif padding_type == "Replikasi":
            result = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REPLICATE)
        elif padding_type == "Refleksi":
            result = cv2.copyMakeBorder(image, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_REFLECT)
        display_before_after(image, result, f"Padding {padding_type}")

def apply_filter(image):
    filter_type = st.selectbox("Tipe Filter", ["Gaussian Blur", "Median Blur", "Bilateral Filter"])
    kernel_size = st.slider("Ukuran Kernel", 3, 15, 5, step=2)  # Hanya bilangan ganjil
    if st.button("Terapkan Filter"):
        if filter_type == "Gaussian Blur":
            result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        elif filter_type == "Median Blur":
            result = cv2.medianBlur(image, kernel_size)
        elif filter_type == "Bilateral Filter":
            result = cv2.bilateralFilter(image, kernel_size, 75, 75)
        display_before_after(image, result, f"{filter_type}")

def advanced_processing_page():
    st.title("Pemrosesan Gambar Lanjutan")
    if 'uploaded_image' not in st.session_state:
        st.write("Tidak ada gambar yang diunggah. Silakan unggah gambar di halaman Beranda.")
        return
    image = st.session_state['uploaded_image']
    operation = st.selectbox("Pilih Operasi", ["Konvolusi", "Padding", "Filter"])
    
    if operation == "Konvolusi":
        apply_convolution(image)
    elif operation == "Padding":
        apply_padding(image)
    elif operation == "Filter":
        apply_filter(image)

def face_detection_page():
    st.title(FACE_DETECTION_TEXT)
    if 'uploaded_image' not in st.session_state:
        st.write("Tidak ada gambar yang diunggah. Silakan unggah gambar di halaman Beranda.")
        return
    image = st.session_state['uploaded_image']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 5)
    processed_image = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(processed_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    display_before_after(image, processed_image, FACE_DETECTION_TEXT)

def contour_analysis_page():
    st.title("Analisis Kontur & Bentuk")
    if 'uploaded_image' not in st.session_state:
        st.write("Tidak ada gambar yang diunggah. Silakan unggah gambar di halaman Beranda.")
        return
    image = st.session_state['uploaded_image']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    display_before_after(image, edges, "Deteksi Tepi Canny")

def compression_comparison_page():
    st.title(COMPRESSION_COMPARISON_TEXT)
    if 'uploaded_image' not in st.session_state:
        st.write("Tidak ada gambar yang diunggah. Silakan unggah gambar di halaman Beranda.")
        return
    image = st.session_state['uploaded_image']
    st.image(image, caption="Gambar Asli", use_column_width=True)
    # Placeholder untuk logika perbandingan kompresi

def color_space_conversion_page():
    st.title(COLOR_SPACE_CONVERSIONS_TEXT)
    if 'uploaded_image' not in st.session_state:
        st.write("Tidak ada gambar yang diunggah. Silakan unggah gambar di halaman Beranda.")
        return
    image = st.session_state['uploaded_image']
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    conversion = st.selectbox("Pilih Konversi", ["RGB", "XYZ", "Lab", "YCbCr", "HSV"])
    if conversion == "RGB":
        display_before_after(image, rgb, "Gambar RGB")
    elif conversion == "XYZ":
        xyz = cv2.cvtColor(rgb, cv2.COLOR_RGB2XYZ)
        display_before_after(image, xyz, "Gambar XYZ")
    elif conversion == "Lab":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        display_before_after(image, lab, "Gambar Lab")
    elif conversion == "YCbCr":
        ycbcr = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        display_before_after(image, ycbcr, "Gambar YCbCr")
    elif conversion == "HSV":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        display_before_after(image, hsv, "Gambar HSV")

def texture_analysis_page():
    st.title(TEXTURE_ANALYSIS_TEXT)
    if 'uploaded_image' not in st.session_state:
        st.write("Tidak ada gambar yang diunggah. Silakan unggah gambar di halaman Beranda.")
        return
    image = st.session_state['uploaded_image']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 24, 3, method="uniform")
    lbp_uint8 = (lbp / lbp.max() * 255).astype(np.uint8)
    display_before_after(image, lbp_uint8, "Tekstur LBP")

class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.landmarks = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                self.landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in hand_landmarks.landmark]
        return frame.from_ndarray(img, format="bgr24")

def gesture_control_drone_page():
    st.title(GESTURE_CONTROL_DRONE_TEXT)
    if model is None:
        st.error("Model tidak dimuat. Tidak dapat melakukan pengenalan gerakan.")
        return
    
    if 'drone' not in st.session_state:
        st.session_state.drone = DroneSimulator()
    
    webrtc_ctx = webrtc_streamer(key="gesture_control", video_processor_factory=GestureProcessor)
    
    if webrtc_ctx.video_processor and webrtc_ctx.video_processor.landmarks:
        landmarks = webrtc_ctx.video_processor.landmarks
        # Ekstrak fitur
        features = []
        for lm in landmarks:
            features.extend([lm['x'], lm['y'], lm['z']])
        features = np.array(features).reshape(1, -1)
        # Prediksi gerakan
        predicted_gesture = model.predict(features)[0]
        st.write(f"Gerakan yang Diprediksi: {predicted_gesture}")
        
        # Pemetaan nama gerakan ke ID
        drone = st.session_state.drone
        gesture_id_map = {v: k for k, v in drone.gesture_labels.items()}
        gesture_id = gesture_id_map.get(predicted_gesture, drone.GESTURE_NONE)
        
        # Perbarui status drone
        state = drone.update_state_from_gesture(gesture_id)
        
        # Tampilkan status drone
        st.write(f"Posisi Drone: {state['posisi']}")
        st.write(f"Rotasi Drone: {state['rotasi']}")
        st.write(f"Terbang: {state['terbang']}")

# Aplikasi Utama
st.set_page_config(page_title="Aplikasi Pemrosesan Gambar", layout="wide")
st.sidebar.title("Fitur")
page = st.sidebar.selectbox("Pilih Fitur", [
    "Beranda",
    "Pemrosesan Gambar",
    "Pemrosesan Lanjutan",
    FACE_DETECTION_TEXT,
    "Analisis Kontur",
    COMPRESSION_COMPARISON_TEXT,
    COLOR_SPACE_CONVERSIONS_TEXT,
    TEXTURE_ANALYSIS_TEXT,
    GESTURE_CONTROL_DRONE_TEXT
])

if page == "Beranda":
    home_page()
elif page == "Pemrosesan Gambar":
    image_processing_page()
elif page == "Pemrosesan Lanjutan":
    advanced_processing_page()
elif page == FACE_DETECTION_TEXT:
    face_detection_page()
elif page == "Analisis Kontur":
    contour_analysis_page()
elif page == COMPRESSION_COMPARISON_TEXT:
    compression_comparison_page()
elif page == COLOR_SPACE_CONVERSIONS_TEXT:
    color_space_conversion_page()
elif page == TEXTURE_ANALYSIS_TEXT:
    texture_analysis_page()
elif page == GESTURE_CONTROL_DRONE_TEXT:
    gesture_control_drone_page()