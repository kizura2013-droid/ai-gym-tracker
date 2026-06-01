import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Setel konfigurasi halaman web agar pas di layar HP
st.set_page_config(page_title="AI Gym Tracker Live", page_icon="🏋️‍♂️", layout="centered")

st.title("🏋️‍♂️ AI GYM TRACKER (LIVE)")
st.caption("Bicep & Tricep Real-time Otomatis Lewat Kamera HP")

# Input Menu Latihan
nama_alat = st.selectbox("Pilih Alat Latihan:", [
    "Dumbbell Bicep Curl", "Barbell Bicep Curl", "EZ Bar Curl", "Cable Bicep Curl",
    "Cable Tricep Pushdown", "Dumbbell Tricep Extension", "Skull Crusher Tricep"
])
berat_beban = st.number_input("Berat Beban (KG):", min_value=1, max_value=200, value=10)

# Gunakan session state khusus agar data sinkron antara video stream dan halaman web
if 'counter' not in st.session_state:
    st.session_state.counter = 0

is_tricep = "Tricep" in nama_alat

class GymVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        # Status internal di dalam class agar tidak terpengaruh refresh server
        self.stage_right = None
        self.stage_left = None
        self.local_counter = 0

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # Efek cermin & konversi warna
        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)

        try:
            landmarks = results.pose_landmarks.landmark
            
            # ---- TANGAN KANAN ----
            r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            angle_right = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
            cv2.putText(img, f"R: {int(angle_right)} deg", tuple(np.multiply(r_elbow, [w, h]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            
            if is_tricep:
                if angle_right < 90: self.stage_right = "down"
                if angle_right > 160 and self.stage_right == 'down':
                    self.stage_right = "up"
                    self.local_counter += 1
            else:
                if angle_right > 160: self.stage_right = "down"
                if angle_right < 30 and self.stage_right == 'down':
                    self.stage_right = "up"
                    self.local_counter += 1

            # ---- TANGAN KIRI ----
            l_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            angle_left = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
            cv2.putText(img, f"L: {int(angle_left)} deg", tuple(np.multiply(l_elbow, [w, h]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
            
            if is_tricep:
                if angle_left < 90: self.stage_left = "down"
                if angle_left > 160 and self.stage_left == 'down':
                    self.stage_left = "up"
                    self.local_counter += 1
            else:
                if angle_left > 160: self.stage_left = "down"
                if angle_left < 30 and self.stage_left == 'down':
                    self.stage_left = "up"
                    self.local_counter += 1

        except:
            pass

        # Gambar teks total reps langsung di dalam video stream (Anti-Delay & Akurat)
        cv2.rectangle(img, (0, 0), (250, 60), (45, 45, 45), -1)
        cv2.putText(img, f"REPS: {self.local_counter}", (15, 42), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(img, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame.from_ndarray(img, format="bgr24")

# Panggil kamera live di halaman web
webrtc_streamer(key="gym-tracker-live", video_processor_factory=GymVideoProcessor)

st.info("💡 Tips: Angka repetisi dicetak langsung di pojok kiri atas dalam kotak video agar hitungannya responsif dan tidak macet!")
