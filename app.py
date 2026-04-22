import streamlit as st
import numpy as np
import wave
import tempfile
import matplotlib.pyplot as plt
import joblib
import datetime

from scipy.signal import find_peaks
import scipy.fft

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# UI
# =========================
st.set_page_config(page_title="Voice AI Stable", layout="centered")
st.title("🎤 Voice AI Analyzer - Stable Production")


# =========================
# Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# SAFE WAV READER (NO LIBROSA, NO SOUNDFILE)
# =========================
def read_wav(file_path):
    with wave.open(file_path, "rb") as wf:
        n_channels = wf.getnchannels()
        frames = wf.getnframes()
        audio = wf.readframes(frames)
        sr = wf.getframerate()

    y = np.frombuffer(audio, dtype=np.int16)

    if n_channels > 1:
        y = y.reshape(-1, n_channels)
        y = np.mean(y, axis=1)

    y = y.astype(np.float32) / 32768.0

    return y, sr


# =========================
# Analysis (NO LIBROSA)
# =========================
def analyze(y, sr):

    energy = np.mean(np.abs(y))
    zcr = np.mean(np.diff(np.sign(y)) != 0)
    shimmer = np.std(y)

    peaks, _ = find_peaks(y, height=0.02)
    speech_rate = len(peaks) / (len(y) / sr)

    pitch_std = np.std(y)

    return {
        "pitch_std": float(pitch_std),
        "energy": float(energy),
        "zcr": float(zcr),
        "shimmer": float(shimmer),
        "speech_rate": float(speech_rate),
    }


# =========================
# Risk
# =========================
def risk_score(f):
    score = 0
    if f["pitch_std"] > 0.05:
        score += 20
    if f["shimmer"] > 0.1:
        score += 20
    if f["speech_rate"] < 2:
        score += 20
    if f["zcr"] < 0.05:
        score += 10
    return min(score, 100)


# =========================
# ML
# =========================
def predict(features):
    X = np.array([[features["pitch_std"], features["energy"],
                   features["zcr"], features["shimmer"],
                   features["speech_rate"]]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    labels = ["طبيعي", "اضطراب صوتي", "مؤشرات تعاطي"]
    return labels[pred], prob


# =========================
# WAV CONVERTER (UPLOAD SAFE)
# =========================
def save_upload(uploaded_file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


# =========================
# Plot
# =========================
def plot_wave(y):
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Waveform")
    return fig


# =========================
# PDF
# =========================
def create_pdf(features, risk, label, img_path):
    file_name = f"report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Voice AI Report", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Risk Score: {risk}%", styles["Heading2"]))
    content.append(Paragraph(f"Result: {label}", styles["Heading2"]))

    content.append(Spacer(1, 10))

    for k, v in features.items():
        content.append(Paragraph(f"{k}: {round(v, 4)}", styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Image(img_path, width=400, height=150))

    doc.build(content)
    return file_name


# =========================
# Session
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# =========================
# Upload ONLY (Stable)
# =========================
uploaded = st.file_uploader(
    "📁 ارفع ملف صوت (WAV فقط لضمان الاستقرار)",
    type=["wav"]
)


if uploaded:

    path = save_upload(uploaded)

    st.session_state.audio_path = path
    st.session_state.audio_bytes = uploaded.read()

    st.success("✔ تم تحميل الملف بنجاح")


# =========================
# PROCESS
# =========================
if st.session_state.audio_path:

    y, sr = read_wav(st.session_state.audio_path)

    features = analyze(y, sr)
    risk = risk_score(features)

    st.subheader("📊 Risk Score")
    st.progress(risk / 100)
    st.write(f"{risk}%")

    st.subheader("🧠 Prediction")

    label, prob = predict(features)

    st.success(label)

    st.write({
        "طبيعي": float(prob[0]),
        "اضطراب": float(prob[1]),
        "تعاطي": float(prob[2]),
    })

    st.audio(st.session_state.audio_bytes)

    st.pyplot(plot_wave(y))

    st.write("### Features")
    st.json(features)

    # =========================
    # PDF
    # =========================
    if st.button("📄 Generate Report"):

        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

        plt.figure()
        plt.plot(y)
        plt.savefig(img_path)
        plt.close()

        pdf = create_pdf(features, risk, label, img_path)

        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf)
