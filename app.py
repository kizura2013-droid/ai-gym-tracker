import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Setel konfigurasi halaman web agar pas di layar HP
st.set_page_config(page_title="AI Gym Tracker", page_icon="🏋️‍♂️", layout="centered")

# Inisialisasi MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# === TAMPILAN INTERFACE WEB ===
st.title("🏋️‍♂️ AI GYM TRACKER")
st.caption("Bicep & Tricep Real-time Web Analyzer")

# Form Input di Web (Otomatis Bagus & Cocok untuk Layar HP)
nama_alat = st.selectbox("Pilih Alat Latihan:", [
    "Dumbbell Bicep Curl", "Barbell Bicep Curl", "EZ Bar Curl", "Cable Bicep Curl",
    "Cable Tricep Pushdown", "Dumbbell Tricep Extension", "Skull Crusher Tricep"
])
berat_beban = st.number_input("Berat Beban (KG):", min_value=1, max_value=200, value=10)

# Tombol untuk Menyalakan Kamera
Mulai = st.checkbox("NYALAKAN AI KAMERA")

# Slot Kosong di Web untuk Menampilkan Video Real-time
FRAME_WINDOW = st.image([])

# Status Latihan (Gunakan Session State agar data tidak hilang saat web refresh)
if 'counter' not in st.session_state:
    st.session_state.counter = 0
if 'stage_right' not in st.session_state:
    st.session_state.stage_right = None
if 'stage_left' not in st.session_state:
    st.session_state.stage_left = None

# Tampilan Status Repetisi di Atas Video
col1, col2 = st.columns(2)
with col1:
    reps_text = st.empty()
with col2:
    feedback_text = st.empty()

is_tricep = "Tricep" in nama_alat

# === LOGIKA UTAMA JIKA KAMERA DINYALAKAN ===
if Mulai:
    # 0 berarti webcam bawaan laptop, jika di hosting internet otomatis mendeteksi kamera HP/Laptop user
    cap = cv2.VideoCapture(0)
    
    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened() and Mulai:
            ret, frame = cap.read()
            if not ret:
                st.write("Gagal membaca kamera.")
                break

            h, w, _ = frame.shape
            # Konversi warna untuk MediaPipe
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            
            feedback = "Mulai Latihan"

            try:
                landmarks = results.pose_landmarks.landmark
                
                # ---- TANGAN KANAN ----
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                
                angle_right = calculate_angle(r_shoulder, r_elbow, r_wrist)
                cv2.putText(image, str(int(angle_right)), tuple(np.multiply(r_elbow, [w, h]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                
                if is_tricep:
                    if angle_right < 90: st.session_state.stage_right = "down"
                    if angle_right > 160 and st.session_state.stage_right == 'down':
                        st.session_state.stage_right = "up"
                        st.session_state.counter += 1
                        feedback = "Tricep Kanan Masuk!"
                else:
                    if angle_right > 160: st.session_state.stage_right = "down"
                    if angle_right < 30 and st.session_state.stage_right == 'down':
                        st.session_state.stage_right = "up"
                        st.session_state.counter += 1
                        feedback = "Bicep Kanan Masuk!"

                # ---- TANGAN KIRI ----
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                angle_left = calculate_angle(l_shoulder, l_elbow, l_wrist)
                cv2.putText(image, str(int(angle_left)), tuple(np.multiply(l_elbow, [w, h]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
                
                if is_tricep:
                    if angle_left < 90: st.session_state.stage_left = "down"
                    if angle_left > 160 and st.session_state.stage_left == 'down':
                        st.session_state.stage_left = "up"
                        st.session_state.counter += 1
                        feedback = "Tricep Kiri Masuk!"
                else:
                    if angle_left > 160: st.session_state.stage_left = "down"
                    if angle_left < 30 and st.session_state.stage_left == 'down':
                        st.session_state.stage_left = "up"
                        st.session_state.counter += 1
                        feedback = "Bicep Kiri Masuk!"

            except:
                feedback = "Tubuh tidak terlihat"
                pass
            
            # Gambar kerangka tubuh
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Update Teks Informasi Real-time di Web
            reps_text.metric(label="TOTAL REPS", value=st.session_state.counter)
            feedback_text.metric(label="FEEDBACK", value=feedback)
            
            # Masukkan hasil gambar kamera ke dalam window Streamlit web
            FRAME_WINDOW.image(image)
    else:
        st.info("Centang kotak di atas untuk mulai mendeteksi gerakan lewat webcam.")