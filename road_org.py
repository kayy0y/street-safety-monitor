import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import random

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Street Safety System", page_icon="🛡️", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.big-title {font-size: 2.2rem; font-weight: bold; color: #1f77b4; text-align: center;}
.alert-high {background: #ffebee; border-left: 5px solid #f44336; padding: 10px; border-radius: 6px;}
.alert-medium {background: #fff8e1; border-left: 5px solid #ff9800; padding: 10px; border-radius: 6px;}
.alert-low {background: #e3f2fd; border-left: 5px solid #2196f3; padding: 10px; border-radius: 6px;}
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "stats" not in st.session_state:
    st.session_state.stats = {"detections": 0, "critical": 0, "medium": 0}
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "prev_frame" not in st.session_state:
    st.session_state.prev_frame = None
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "paused" not in st.session_state:
    st.session_state.paused = False

# ---------------- DETECTOR SETUP ----------------
HAAR_BASE = cv2.data.haarcascades
PERSON_CASCADE_PATH = HAAR_BASE + "haarcascade_fullbody.xml"
FACE_CASCADE_PATH = HAAR_BASE + "haarcascade_frontalface_default.xml"

person_cascade = cv2.CascadeClassifier(PERSON_CASCADE_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# ---------------- DETECTION FUNCTIONS ----------------
def detect_people(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return person_cascade.detectMultiScale(gray, 1.1, 3)

def blur_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]
        if roi.size:
            frame[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (99, 99), 30)
    return frame

def detect_motion(frame, prev):
    if prev is None:
        return 0
    gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return np.sum(thresh) / 255

def generate_alerts(frame_count, count, motion):
    alerts = []
    if frame_count % 80 == 0 and count >= 2:
        alerts.append({
            "type": "Following Pattern", "level": "medium",
            "message": "Multiple people detected close together",
            "confidence": 82, "timestamp": datetime.now().strftime("%H:%M:%S"),
            "location": f"Street Light {random.randint(1, 10)}"
        })
    if motion > 60000:
        alerts.append({
            "type": "Rapid Movement", "level": "high",
            "message": "Sudden rapid movement detected",
            "confidence": 90, "timestamp": datetime.now().strftime("%H:%M:%S"),
            "location": f"Street Light {random.randint(1, 10)}"
        })
    if frame_count % 120 == 0 and count == 1:
        alerts.append({
            "type": "Person Alone", "level": "low",
            "message": "Single person detected",
            "confidence": 75, "timestamp": datetime.now().strftime("%H:%M:%S"),
            "location": f"Street Light {random.randint(1, 10)}"
        })
    return alerts

def annotate_frame(frame, people, motion):
    for (x, y, w, h) in people:
        color = (0, 255, 0) if motion <= 60000 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "Person", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"People: {len(people)} | Motion: {int(motion/1000)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

# ---------------- MAIN APP ----------------
def main():
    st.markdown('<p class="big-title">🛡️ Street Safety Monitoring</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#777;'>Live webcam analysis</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Webcam Feed")
        frame_placeholder = st.empty()

        start = st.button("▶️ Start Camera")
        stop = st.button("⏹️ Stop Camera")
        pause = st.button("⏸️ Pause/Resume")

        if start:
            st.session_state.camera_running = True
            st.session_state.paused = False

        if stop:
            st.session_state.camera_running = False
            st.session_state.prev_frame = None

        if pause:
            st.session_state.paused = not st.session_state.paused

        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break
                if st.session_state.paused:
                    frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                    continue

                st.session_state.frame_count += 1

                people = detect_people(frame)
                motion = detect_motion(frame, st.session_state.prev_frame)
                st.session_state.prev_frame = frame.copy()

                alerts = generate_alerts(st.session_state.frame_count, len(people), motion)
                for alert in alerts:
                    st.session_state.alerts.insert(0, alert)
                    if alert["level"] == "high":
                        st.session_state.stats["critical"] += 1
                    elif alert["level"] == "medium":
                        st.session_state.stats["medium"] += 1
                st.session_state.stats["detections"] += len(people)
                st.session_state.alerts = st.session_state.alerts[:15]

                frame = blur_faces(frame)
                frame = annotate_frame(frame, people, motion)
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)

            cap.release()

    with col2:
        st.subheader("🚨 Alerts")
        if not st.session_state.alerts:
            st.success("✅ No alerts right now.")
        else:
            for alert in st.session_state.alerts[:8]:
                alert_class = f"alert-{alert['level']}"
                icon = "🚨" if alert["level"] == "high" else "⚠️" if alert["level"] == "medium" else "ℹ️"
                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>{icon} {alert['type']}</strong><br>
                    {alert['message']}<br>
                    <small>📍 {alert['location']} | ⏰ {alert['timestamp']} | 🎯 {alert['confidence']}%</small>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        st.metric("Detections", st.session_state.stats["detections"])
        st.metric("Critical Alerts", st.session_state.stats["critical"])
        st.metric("Medium Alerts", st.session_state.stats["medium"])

        if st.button("🔄 Reset"):
            st.session_state.alerts.clear()
            st.session_state.stats = {"detections": 0, "critical": 0, "medium": 0}
            st.session_state.frame_count = 0
            st.session_state.prev_frame = None
            st.session_state.camera_running = False
            st.session_state.paused = False
            st.experimental_rerun()

if __name__ == "__main__":
    main()
