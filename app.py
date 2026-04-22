import streamlit as st
import numpy as np
import librosa
import tempfile
import joblib
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import find_peaks
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# UI
# =========================
st.set_page_config(page_title="Voice AI Stable", layout="centered")
st.title("🎤 Voice AI Analyzer (Stable Production)")


# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# LOAD AUDIO SAFE (ANY FORMAT)
# =========================
def load_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    y, sr = librosa.load(tmp_path, sr=16000, mono=True)
    return y, sr, tmp_path


# =========================
# FEATURES
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
# FIXED PREDICTION
# =========================
def predict(features):
    X = np.array([[
        features["pitch_std"],
        features["energy"],
        features["zcr"],
        features["shimmer"],
        features["speech_rate"]
    ]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    labels = ["طبيعي", "اضطراب صوتي", "مؤشرات تعاطي"]
    return labels[pred], prob


# =========================
# PLOT
# =========================
def plot_wave(y):
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Waveform")
    return fig


# =========================
# PDF
# =========================
def create_pdf(features, label):
    name = f"report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    doc = SimpleDocTemplate(name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Voice Report", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Result: {label}", styles["Heading2"]))

    for k, v in features.items():
        content.append(Paragraph(f"{k}: {v:.4f}", styles["Normal"]))

    doc.build(content)
    return name


# =========================
# UPLOAD (iPhone SAFE)
# =========================
uploaded = st.file_uploader(
    "📁 Upload audio/video (iPhone supported)",
    type=["wav", "mp3", "m4a", "mp4", "mov"]
)

if uploaded:

    y, sr, path = load_audio(uploaded)

    features = analyze(y, sr)
    label, prob = predict(features)

    st.subheader("📊 Result")
    st.success(label)

    st.write({
        "طبيعي": float(prob[0]),
        "اضطراب": float(prob[1]),
        "تعاطي": float(prob[2]),
    })

    st.audio(uploaded)

    st.pyplot(plot_wave(y))

    st.write("### Features")
    st.json(features)

    if st.button("📄 PDF Report"):
        pdf = create_pdf(features, label)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf)
