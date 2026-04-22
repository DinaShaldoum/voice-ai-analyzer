import streamlit as st
import numpy as np
import tempfile
import datetime
import joblib
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
import wave
import io

from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av


# =========================
# UI
# =========================
st.set_page_config(page_title="Voice AI iPhone Ready", layout="centered")
st.title("🎤 Voice AI Analyzer (iPhone Stable)")


# =========================
# MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("voice_model.pkl")

model = load_model()


# =========================
# SAFE WAV READER (NO LIBROSA)
# =========================
def read_wav(file_path):
    with wave.open(file_path, "rb") as wf:
        sr = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    y = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
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
# AUDIO PROCESSOR (iPhone Safe)
# =========================
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame):
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame


# =========================
# SESSION
# =========================
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None


# =========================
# INPUT MODE
# =========================
mode = st.radio("اختر:", ["📁 رفع ملف صوت", "🎙️ تسجيل من iPhone"])


# =========================
# UPLOAD
# =========================
if mode == "📁 رفع ملف صوت":
    uploaded = st.file_uploader("ارفع WAV فقط", type=["wav"])

    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(uploaded.read())
        tmp.close()

        st.session_state.audio_path = tmp.name
        st.session_state.audio_bytes = open(tmp.name, "rb").read()

        st.success("تم رفع الملف ✔")


# =========================
# iPhone RECORD
# =========================
else:
    st.subheader("🎙️ تسجيل صوت من iPhone")

    ctx = webrtc_streamer(
        key="iphone-audio",
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
    )

    if ctx.audio_processor:
        if st.button("💾 حفظ التسجيل"):
            audio = np.concatenate(ctx.audio_processor.frames, axis=1)

            audio = audio.T.astype(np.float32)

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

            import soundfile as sf
            sf.write(tmp.name, audio, 44100)

            st.session_state.audio_path = tmp.name
            st.session_state.audio_bytes = open(tmp.name, "rb").read()

            st.success("تم حفظ التسجيل ✔")


# =========================
# PROCESS
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
