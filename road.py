
import streamlit as st
import cv2
import numpy as np
import random
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import os

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
if "running" not in st.session_state:
    st.session_state.running = False
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "prev_frame" not in st.session_state:
    st.session_state.prev_frame = None

# ---------------- DETECTOR UTILITIES ----------------
HAAR_BASE = cv2.data.haarcascades

# Ensure cascade files exist; if not, the app warns but continues
PERSON_CASCADE_PATH = os.path.join(HAAR_BASE, "haarcascade_fullbody.xml")
FACE_CASCADE_PATH = os.path.join(HAAR_BASE, "haarcascade_frontalface_default.xml")

if not os.path.exists(PERSON_CASCADE_PATH) or not os.path.exists(FACE_CASCADE_PATH):
    st.warning("Haarcascade xml files not found in OpenCV install. Person/face detection might fail.")
person_cascade = cv2.CascadeClassifier(PERSON_CASCADE_PATH)
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

def detect_people(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    people = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    return people

def blur_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
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

# ---------------- RTC CONFIG (updated TURN server) ----------------
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # public metered TURN - replace with your TURN if you have one for production
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject"
            }
        ]
    }
)

# ---------------- video frame callback (runs in main thread) ----------------
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This callback is executed in the main Streamlit thread (safe to manipulate session_state).
    It receives an av.VideoFrame and must return an av.VideoFrame.
    """
    frm = frame.to_ndarray(format="bgr24")
    st.session_state.frame_count += 1

    # Do detection
    people = ()
    try:
        people = detect_people(frm)
    except Exception:
        # If cascade empty or detection fails, ignore and continue
        people = ()

    motion = detect_motion(frm, st.session_state.prev_frame) if st.session_state.prev_frame is not None else 0
    st.session_state.prev_frame = frm.copy()

    # Generate alerts and update session state (safe here)
    alerts = generate_alerts(st.session_state.frame_count, len(people), motion)
    for alert in alerts:
        st.session_state.alerts.insert(0, alert)
        if alert["level"] == "high":
            st.session_state.stats["critical"] += 1
        elif alert["level"] == "medium":
            st.session_state.stats["medium"] += 1

    st.session_state.stats["detections"] += len(people)
    # Keep only the latest 15 alerts
    st.session_state.alerts = st.session_state.alerts[:15]

    # Blur faces and annotate
    try:
        frm = blur_faces(frm)
    except Exception:
        pass
    frm = annotate_frame(frm, people, motion)

    # Return modified frame
    return av.VideoFrame.from_ndarray(frm, format="bgr24")

# ---------------- MAIN APP ----------------
def main():
    st.markdown('<p class="big-title">🛡️ Street Safety Monitoring</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#777;'>AI-Powered Safety Detection</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Feed")

        # Start/stop toggle button
        if st.button("▶️ Start" if not st.session_state.running else "⏸️ Stop"):
            st.session_state.running = not st.session_state.running

        status_placeholder = st.empty()

        if st.session_state.running:
            try:
                webrtc_ctx = webrtc_streamer(
                    key="live-stream",
                    mode="sendrecv",
                    rtc_configuration=rtc_configuration,
                    video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                    async_processing=False,
                )

                # show friendly status
                if webrtc_ctx and webrtc_ctx.state.playing:
                    status_placeholder.success("🟢 Camera Active — allow camera access in your browser if prompted.")
                else:
                    status_placeholder.info("Waiting for camera... please allow camera access in the browser.")
            except Exception as e:
                status_placeholder.error(f"Camera error: {e}")
                st.session_state.running = False
        else:
            status_placeholder.info("⏸️ Camera Paused")
            demo_img = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(demo_img, "Click 'Start' to open camera",
                        (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            st.image(demo_img, channels="BGR", use_container_width=True)

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
            st.experimental_rerun()

if __name__ == "__main__":
    main()
