import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import tempfile
import datetime
import joblib
import subprocess
from scipy.signal import find_peaks
import scipy.fft

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# إعداد الصفحة
# =========================
st.set_page_config(page_title="Voice AI Analyzer PRO", layout="centered")
st.title("🎤 Voice AI Analyzer - Production Version")


# =========================
# تحميل الموديل
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# استخراج الصوت من فيديو (FFmpeg)
# =========================
def extract_audio(file_path):
    audio_path = tempfile.mktemp(suffix=".wav")

    command = [
        "ffmpeg",
        "-y",
        "-i", file_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "22050",
        "-ac", "1",
        audio_path
    ]

    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return audio_path


# =========================
# تحميل صوت آمن (بدون librosa.load)
# =========================
def load_audio_safe(path):
    y, sr = sf.read(path)

    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    sr = 22050
    return y, sr


# =========================
# تحليل الصوت
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
# Risk Score
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
# ML Prediction
# =========================
def classify_ml(features):
    X = np.array([[features["pitch_std"], features["energy"],
                   features["zcr"], features["shimmer"],
                   features["speech_rate"]]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    labels = ["طبيعي", "اضطراب صوتي", "مؤشرات تعاطي"]
    return labels[pred], prob


# =========================
# Visualizations
# =========================
def plot_wave(y):
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Waveform")
    return fig


def plot_frequency(y, sr):
    fft = np.abs(scipy.fft.fft(y))
    freq = scipy.fft.fftfreq(len(fft), 1/sr)

    fig, ax = plt.subplots()
    ax.plot(freq[:len(freq)//2], fft[:len(freq)//2])
    ax.set_title("Frequency Spectrum")
    return fig


# =========================
# PDF Report
# =========================
def create_pdf(features, risk, label, img_path):
    file_name = f"report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Voice AI Analysis Report", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Risk Score: {risk}%", styles["Heading2"]))
    content.append(Paragraph(f"Prediction: {label}", styles["Heading2"]))
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
# UI
# =========================
st.subheader("📁 رفع فيديو أو صوت")

uploaded = st.file_uploader(
    "ارفع ملف (iPad / iPhone / MP4 / MOV / MP3 / WAV)",
    type=["mp4", "mov", "mp3", "wav", "m4a"]
)


# =========================
# معالجة الملف
# =========================
if uploaded:

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(uploaded.read())
    tmp_file.close()

    audio_path = extract_audio(tmp_file.name)

    st.session_state.audio_path = audio_path
    st.session_state.audio_bytes = open(audio_path, "rb").read()

    st.success("✔ تم استخراج الصوت بنجاح")


# =========================
# التحليل
# =========================
if st.session_state.audio_path:

    y, sr = load_audio_safe(st.session_state.audio_path)

    features = analyze_audio(y, sr)
    risk = risk_score(features)

    st.subheader("📊 Risk Score")
    st.progress(risk / 100)
    st.write(f"{risk}%")

    st.subheader("🧠 ML Result")
    label, prob = classify_ml(features)

    st.success(label)

    st.write({
        "طبيعي": float(prob[0]),
        "اضطراب": float(prob[1]),
        "تعاطي": float(prob[2]),
    })

    st.audio(st.session_state.audio_bytes)

    st.pyplot(plot_wave(y))
    st.pyplot(plot_frequency(y, sr))

    st.write("### Features")
    st.json(features)

    # =========================
    # PDF
    # =========================
    if st.button("📄 إنشاء تقرير PDF"):

        img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

        plt.figure()
        plt.plot(y)
        plt.savefig(img_path)
        plt.close()

        pdf_file = create_pdf(features, risk, label, img_path)

        with open(pdf_file, "rb") as f:
            st.download_button("تحميل التقرير", f, file_name=pdf_file)
