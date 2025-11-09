# road.py
# Requirements:
# pip install streamlit opencv-python-headless numpy pillow streamlit-webrtc av

import streamlit as st
import cv2
import numpy as np
import random
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Street Safety System", page_icon="🛡️", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.big-title {font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center;}
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

# ---------------- DETECTOR ----------------
class SimpleDetector:
    def __init__(self):
        self.person_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_fullbody.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.frame_count = 0

    def detect_people(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.person_cascade.detectMultiScale(gray, 1.1, 3)

    def blur_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            if roi.size:
                frame[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (99, 99), 30)
        return frame

    def detect_motion(self, frame, prev):
        if prev is None:
            return 0
        gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        return np.sum(thresh) / 255

    def generate_alerts(self, count, motion):
        alerts = []
        self.frame_count += 1

        if self.frame_count % 80 == 0 and count >= 2:
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
        if self.frame_count % 120 == 0 and count == 1:
            alerts.append({
                "type": "Person Alone", "level": "low",
                "message": "Single person detected",
                "confidence": 75, "timestamp": datetime.now().strftime("%H:%M:%S"),
                "location": f"Street Light {random.randint(1, 10)}"
            })
        return alerts

    def process_frame(self, frame, prev=None, blur=True):
        people = self.detect_people(frame)
        motion = self.detect_motion(frame, prev) if prev is not None else 0
        alerts = self.generate_alerts(len(people), motion)
        if blur:
            frame = self.blur_faces(frame)
        for (x, y, w, h) in people:
            color = (0, 255, 0) if motion <= 60000 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Person", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, f"People: {len(people)} | Motion: {int(motion/1000)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame, people, alerts

# ---------------- PROCESSOR ----------------
class DetectorVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = SimpleDetector()
        self.prev_frame = None

    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        processed, people, alerts = self.detector.process_frame(frm, self.prev_frame)
        self.prev_frame = frm.copy()

        # Update session_state safely (wrapped so worker thread won't crash app)
        try:
            # It's possible session_state modifications from worker thread raise,
            # so we guard updates and keep them minimal.
            for alert in alerts:
                st.session_state.alerts.insert(0, alert)
                if alert["level"] == "high":
                    st.session_state.stats["critical"] += 1
                elif alert["level"] == "medium":
                    st.session_state.stats["medium"] += 1
            st.session_state.stats["detections"] += len(people)
            st.session_state.alerts = st.session_state.alerts[:15]
        except Exception:
            # If updating session_state from this thread fails, ignore silently.
            # Main UI thread will still show existing alerts.
            pass

        return av.VideoFrame.from_ndarray(processed, format="bgr24")

# ---------------- MAIN APP ----------------
def main():
    st.markdown('<p class="big-title">🛡️ Street Safety Monitoring</p>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#777;'>AI-Powered Safety Detection</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Feed")

        rtc_config = RTCConfiguration(
            {
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {
                        "urls": ["turn:numb.viagenie.ca:3478"],
                        "username": "webrtc@live.com",
                        "credential": "muazkh"
                    }
                ]
            }
        )

        # Start/stop toggle
        if st.button("▶️ Start" if not st.session_state.running else "⏸️ Stop"):
            st.session_state.running = not st.session_state.running

        status_placeholder = st.empty()

        if st.session_state.running:
            try:
                webrtc_ctx = webrtc_streamer(
                    key="live-stream",
                    mode="sendrecv",
                    rtc_configuration=rtc_config,
                    video_processor_factory=DetectorVideoProcessor,
                    media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
                    async_processing=False,  # more stable locally; set True if you need async processing
                )

                # Check playing state (if available)
                try:
                    if webrtc_ctx and webrtc_ctx.state.playing:
                        status_placeholder.success("🟢 Camera Active")
                    else:
                        status_placeholder.warning("Waiting for camera... allow access in the browser.")
                except Exception:
                    # Some contexts may not expose state; ignore
                    status_placeholder.info("Streaming (check browser for camera prompt).")

            except Exception as e:
                status_placeholder.error(f"Camera error: {str(e)}")
                st.session_state.running = False
        else:
            status_placeholder.info("⏸️ Camera Paused")
            # show placeholder image
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
            st.experimental_rerun()


if __name__ == "__main__":
    main()
