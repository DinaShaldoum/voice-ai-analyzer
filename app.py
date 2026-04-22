import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import tempfile
import datetime
import scipy.fft
from scipy.signal import find_peaks
import joblib
import os
import sounddevice as sd

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# إعداد الصفحة
# =========================
st.set_page_config(page_title="Voice AI Analyzer", layout="centered")
st.title("🎤 نظام تحليل الصوت")


# =========================
# تحميل الموديل
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# تحليل الصوت
# =========================
def analyze_audio(y, sr):
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]

    pitch_std = np.std(pitch_values) if len(pitch_values) else 0
    energy = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    shimmer = np.std(librosa.feature.rms(y=y))

    peaks, _ = find_peaks(y, height=0.02)
    speech_rate = len(peaks) / (len(y) / sr)

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
    if f["pitch_std"] > 50:
        score += 20
    if f["shimmer"] > 0.05:
        score += 20
    if f["speech_rate"] < 2:
        score += 20
    if f["zcr"] < 0.05:
        score += 10
    return min(score, 100)


# =========================
# ML
# =========================
def classify_ml(features):
    X = np.array([[features["pitch_std"], features["energy"], features["zcr"],
                   features["shimmer"], features["speech_rate"]]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    labels = ["طبيعي", "اضطراب صوتي", "مؤشرات تعاطي"]
    return labels[pred], prob


# =========================
# الرسوم
# =========================
def plot_wave(y):
    fig, ax = plt.subplots()
    ax.plot(y)
    ax.set_title("Waveform")
    return fig


def plot_frequency(y, sr):
    fft = np.abs(scipy.fft.fft(y))
    freq = scipy.fft.fftfreq(len(fft), 1/sr)
    half = len(freq)//2

    fig, ax = plt.subplots()
    ax.plot(freq[:half], fft[:half])
    ax.set_title("Frequency Spectrum")
    return fig


def plot_spectrogram(y, sr):
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)

    fig, ax = plt.subplots()
    librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    ax.set_title("Spectrogram")
    return fig


# =========================
# PDF
# =========================
def create_pdf(features, risk, label, img_path):
    file_name = f"report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Voice Analysis Report", styles["Title"]))
    content.append(Spacer(1, 10))

    content.append(Paragraph(f"Risk Score: {risk}%", styles["Heading2"]))
    content.append(Paragraph(f"ML Result: {label}", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for k, v in features.items():
        content.append(Paragraph(f"{k}: {round(v, 4)}", styles["Normal"]))

    content.append(Spacer(1, 10))
    content.append(Image(img_path, width=400, height=150))

    doc.build(content)
    return file_name


# =========================
# Session State
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# =========================
# اختيار
# =========================
option = st.radio("اختر:", ["رفع ملف", "تسجيل"])


# =========================
# رفع ملف
# =========================
if option == "رفع ملف":
    uploaded = st.file_uploader("ارفع ملف صوت", type=["wav", "mp3"])

    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(uploaded.read())
        tmp.close()

        st.session_state.audio_path = tmp.name
        st.session_state.audio_bytes = open(tmp.name, "rb").read()


# =========================
# تسجيل صوت (FIXED)
# =========================
else:
    if st.button("🎙️ تسجيل 10 ثواني"):

        sr = 22050
        duration = 10

        rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        rec = rec.squeeze()

        tmp_path = tempfile.mktemp(suffix=".wav")
        sf.write(tmp_path, rec, sr)

        st.session_state.audio_path = tmp_path
        st.session_state.audio_bytes = open(tmp_path, "rb").read()

        st.success("تم التسجيل بنجاح 🎉")


# =========================
# التحليل
# =========================
if st.session_state.audio_path:

    y, sr = librosa.load(st.session_state.audio_path)

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
    st.pyplot(plot_spectrogram(y, sr))

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