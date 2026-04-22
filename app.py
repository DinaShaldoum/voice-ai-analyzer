import streamlit as st
import numpy as np
import tempfile
import joblib
import wave
import matplotlib.pyplot as plt
import datetime
from scipy.signal import find_peaks
import soundfile as sf
import base64
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# UI
# =========================
st.set_page_config(page_title="Voice AI Stable iPhone", layout="centered")
st.title("🎤 Voice AI Analyzer (Stable iPhone Version)")


# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# JS AUDIO RECORDER (iPhone SAFE)
# =========================
def audio_recorder():
    return st.markdown("""
    <audio id="recorder" controls></audio>
    <button onclick="startRecording()">🎙️ Start</button>
    <button onclick="stopRecording()">⏹ Stop</button>
    <script>
    let recorder;
    let chunks = [];

    async function startRecording(){
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        recorder = new MediaRecorder(stream);
        recorder.start();

        recorder.ondataavailable = e => chunks.push(e.data);

        recorder.onstop = async () => {
            const blob = new Blob(chunks, { type: 'audio/wav' });
            const reader = new FileReader();
            reader.readAsDataURL(blob);
            reader.onloadend = () => {
                window.parent.postMessage({
                    type: "audio_data",
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
# RECEIVE AUDIO
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# =========================
# UPLOAD
# =========================
uploaded = st.file_uploader("📁 Upload WAV / MP3", type=["wav", "mp3"])

if uploaded:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.write(uploaded.read())
    tmp.close()

    st.session_state.audio_path = tmp.name
    st.session_state.audio_bytes = open(tmp.name, "rb").read()

    st.success("Uploaded ✔")


# =========================
# RECORD UI
# =========================
st.subheader("🎙️ Record (iPhone Ready)")
audio_recorder()


# =========================
# SAFE WAV LOAD
# =========================
def load_audio(path):
    y, sr = sf.read(path)
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    return y, sr


# =========================
# ANALYSIS
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
# ML
# =========================
def predict(features):
    X = np.array([[features["energy"], features["zcr"],
                   features["shimmer"], features["speech_rate"]]])

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    labels = ["طبيعي", "اضطراب", "مؤشرات"]
    return labels[pred], prob


# =========================
# PDF
# =========================
def create_pdf(features, label):
    file_name = f"report_{datetime.datetime.now().strftime('%H%M%S')}.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Voice AI Report", styles["Title"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Result: {label}", styles["Heading2"]))

    for k, v in features.items():
        content.append(Paragraph(f"{k}: {v:.4f}", styles["Normal"]))

    doc.build(content)
    return file_name


# =========================
# PROCESS
# =========================
if st.session_state.audio_path:

    y, sr = load_audio(st.session_state.audio_path)

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

    fig, ax = plt.subplots()
    ax.plot(y)
    st.pyplot(fig)

    st.write(features)

    if st.button("📄 PDF Report"):
        pdf = create_pdf(features, label)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, file_name=pdf)
