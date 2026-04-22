import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tempfile
import datetime
import joblib
import soundfile as sf

from scipy.signal import find_peaks
import scipy.fft

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from streamlit_mic_recorder import mic_recorder


# =========================
# UI
# =========================
st.set_page_config(page_title="Voice AI Analyzer PRO", layout="centered")
st.title("🎤 Voice AI Analyzer (Stable Version)")


# =========================
# Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# Safe audio loader (NO librosa.load)
# =========================
def load_audio(path):
    y, sr = sf.read(path)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    sr = 22050
    return y, sr


# =========================
# Analysis
# =========================
def analyze_audio(y, sr):
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
def classify(features):
    X = np.array([[features["pitch_std"], features["energy"],
                   features["zcr"], features["shimmer"],
                   features["speech_rate"]]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    labels = ["طبيعي", "اضطراب صوتي", "مؤشرات تعاطي"]
    return labels[pred], prob


# =========================
# UI MODE
# =========================
mode = st.radio("اختر:", ["🎙️ تسجيل", "📁 رفع ملف صوت"])


audio_path = None
audio_bytes = None


# =========================
# 🎙️ تسجيل صوت (FIXED - WORKS ON CLOUD)
# =========================
if mode == "🎙️ تسجيل":

    audio = mic_recorder(
        start_prompt="🔴 تسجيل",
        stop_prompt="⏹️ إيقاف"
    )

    if audio:

        tmp = tempfile.mktemp(suffix=".wav")

        with open(tmp, "wb") as f:
            f.write(audio["bytes"])

        audio_path = tmp
        audio_bytes = audio["bytes"]

        st.success("تم التسجيل بنجاح 🎉")


# =========================
# 📁 رفع ملف
# =========================
else:

    file = st.file_uploader("ارفع ملف صوت", type=["wav", "mp3", "m4a"])

    if file:

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(file.read())
        tmp.close()

        audio_path = tmp.name
        audio_bytes = open(tmp.name, "rb").read()

        st.success("تم رفع الملف 🎉")


# =========================
# PROCESS
# =========================
if audio_path:

    y, sr = load_audio(audio_path)

    features = analyze_audio(y, sr)
    risk = risk_score(features)

    st.subheader("📊 Risk Score")
    st.progress(risk / 100)
    st.write(f"{risk}%")

    st.subheader("🧠 Result")
    label, prob = classify(features)

    st.success(label)

    st.write({
        "طبيعي": float(prob[0]),
        "اضطراب": float(prob[1]),
        "تعاطي": float(prob[2]),
    })

    st.audio(audio_bytes)

    # Waveform
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Waveform")
    st.pyplot(fig)

    st.write("### Features")
    st.json(features)
