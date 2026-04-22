import streamlit as st
import numpy as np
import tempfile
import datetime
import joblib
import matplotlib.pyplot as plt
import wave
import io

from scipy.signal import find_peaks
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av


# =========================
# UI
# =========================
st.set_page_config(page_title="Voice AI Realtime", layout="centered")
st.title("🎤 Voice AI Realtime Analyzer (Stable)")


# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# SAFE WAV READER
# =========================
def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        audio = wf.readframes(wf.getnframes())

    y = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    return y, sr


# =========================
# ANALYSIS ENGINE (stable)
# =========================
def analyze(y, sr):
    energy = np.mean(np.abs(y))
    zcr = np.mean(np.diff(np.sign(y)) != 0)
    shimmer = np.std(y)

    peaks, _ = find_peaks(y, height=0.02)
    speech_rate = len(peaks) / (len(y) / sr)

    return {
        "energy": float(energy),
        "zcr": float(zcr),
        "shimmer": float(shimmer),
        "speech_rate": float(speech_rate),
    }


# =========================
# ML PREDICTION
# =========================
def predict(features):
    X = np.array([[features["energy"], features["zcr"],
                   features["shimmer"], features["speech_rate"]]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    labels = ["طبيعي", "اضطراب", "مؤشرات"]
    return labels[pred], prob


# =========================
# PDF REPORT
# =========================
def create_pdf(features, label, img_path):
    file_name = f"report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Voice AI Report", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Result: {label}", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for k, v in features.items():
        content.append(Paragraph(f"{k}: {round(v, 4)}", styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Image(img_path, width=400, height=150))

    doc.build(content)
    return file_name


# =========================
# REALTIME AUDIO PROCESSOR (FIXED)
# =========================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()

        # safe append (no shape crash)
        if audio is not None:
            self.buffer.append(audio)

        return frame


# =========================
# SESSION STATE
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# =========================
# MODE
# =========================
mode = st.radio("اختر:", ["📁 رفع ملف", "🎙️ تسجيل realtime"])


# =========================
# UPLOAD
# =========================
if mode == "📁 رفع ملف":
    uploaded = st.file_uploader("ارفع WAV فقط", type=["wav"])

    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(uploaded.read())
        tmp.close()

        st.session_state.audio_path = tmp.name
        st.session_state.audio_bytes = open(tmp.name, "rb").read()

        st.success("✔ تم الرفع")


# =========================
# REALTIME RECORD (FIXED)
# =========================
else:
    st.subheader("🎙️ تسجيل realtime من iPhone")

    ctx = webrtc_streamer(
        key="voice",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.audio_processor:

        if st.button("💾 حفظ التسجيل"):

            frames = ctx.audio_processor.buffer

            # 🔥 FIX: منع crash
            clean = [f for f in frames if f is not None]

            if len(clean) == 0:
                st.warning("لا يوجد صوت مسجل")
                st.stop()

            audio = np.hstack(clean)

            audio = audio.T.astype(np.float32)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

            import soundfile as sf
            sf.write(tmp.name, audio, 44100)

            st.session_state.audio_path = tmp.name
            st.session_state.audio_bytes = open(tmp.name, "rb").read()

            st.success("✔ تم الحفظ")


# =========================
# PROCESSING
# =========================
if st.session_state.audio_path:

    y, sr = read_wav(st.session_state.audio_path)

    features = analyze(y, sr)
    label, prob = predict(features)

    st.subheader("📊 Result")
    st.success(label)

    st.write({
        "طبيعي": float(prob[0]),
        "اضطراب": float(prob[1]),
        "مؤشرات": float(prob[2]),
    })

    st.audio(st.session_state.audio_bytes)

    st.write("### Features")
    st.json(features)

    # =========================
    # WAVE
    # =========================
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # =========================
    # PDF
    # =========================
    if st.button("📄 Generate PDF"):

        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

        plt.figure()
        plt.plot(y)
        plt.savefig(img_path)
        plt.close()

        pdf = create_pdf(features, label, img_path)

        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf)
