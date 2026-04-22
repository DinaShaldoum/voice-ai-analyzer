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
import base64

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# PAGE
# =========================
st.set_page_config(page_title="Voice AI iPhone", layout="centered")
st.title("🎤 Voice AI Analyzer (iPhone Stable)")


# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# SAFE AUDIO LOAD
# =========================
def load_audio(file):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(file.read())
    tmp.close()

    y, sr = librosa.load(tmp.name, sr=16000, mono=True)
    return y, sr, tmp.name


# =========================
# FEATURES
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
# ML
# =========================
def classify(features):
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

    content.append(Paragraph("Voice AI Report", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Result: {label}", styles["Heading2"]))

    for k, v in features.items():
        content.append(Paragraph(f"{k}: {v:.4f}", styles["Normal"]))

    doc.build(content)
    return name


# =========================
# 🔴 iPhone Recorder (HTML)
# =========================
def audio_recorder():
    st.markdown("""
    <h3>🎙️ تسجيل مباشر من iPhone</h3>

    <button onclick="startRecording()">Start Recording</button>
    <button onclick="stopRecording()">Stop</button>

    <script>
    let recorder;
    let chunks = [];

    async function startRecording(){
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(stream);
        recorder.start();

        recorder.ondataavailable = e => chunks.push(e.data);

        recorder.onstop = () => {
            const blob = new Blob(chunks, { type: 'audio/wav' });
            const reader = new FileReader();
            reader.readAsDataURL(blob);

            reader.onloadend = () => {
                window.parent.postMessage({
                    type: "streamlit:audio",
                    data: reader.result
                }, "*");
            }
        }
    }

    function stopRecording(){
        recorder.stop();
    }
    </script>
    """, unsafe_allow_html=True)


# =========================
# STATE
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# =========================
# INPUT
# =========================
mode = st.radio("اختر:", ["📁 رفع ملف", "🎙️ تسجيل iPhone"])


# =========================
# UPLOAD
# =========================
if mode == "📁 رفع ملف":
    uploaded = st.file_uploader("Upload audio/video", type=["wav", "mp3", "m4a", "mp4", "mov"])

    if uploaded:
        y, sr, path = load_audio(uploaded)

        st.session_state.audio_path = path
        st.session_state.audio_bytes = uploaded.read()

        st.success("Uploaded ✔")


# =========================
# RECORD UI
# =========================
else:
    audio_recorder()


# =========================
# PROCESS
# =========================
if st.session_state.audio_path:

    y, sr = librosa.load(st.session_state.audio_path, sr=16000)

    features = analyze_audio(y, sr)
    label, prob = classify(features)

    st.subheader("📊 Result")
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

    if st.button("📄 PDF Report"):
        pdf = create_pdf(features, label)
        with open(pdf, "rb") as f:
            st.download_button("Download", f, file_name=pdf)
