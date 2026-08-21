import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import cv2
import tensorflow as tf

# ================= CONFIG =================
MODEL_PATH = "/home/sre/Desktop/Banana/jetson-banana/weevil_classifier_keras.h5"
TEMP_AUDIO = "temp_recording.wav"
CLASSES = ['stage_1','stage_2','stage_3','stage_4','stage_5','no_larvae']
RECORD_DURATION = 10
SAMPLE_RATE = 44100
# ==========================================


# ---------- Load Model ----------
print("🔄 Loading H5 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded")


# ---------- Record Audio ----------
def record_audio():
    print(f"🎙 Recording for {RECORD_DURATION} seconds...")
    recording = sd.rec(
        int(RECORD_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='int16'
    )
    sd.wait()
    sf.write(TEMP_AUDIO, recording, SAMPLE_RATE)
    print("✅ Recording Saved")


# ---------- Preprocess ----------
def preprocess_audio(file_path):
    audio, _ = librosa.load(file_path, sr=16000)

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=16000,
        n_mels=128,
        fmax=2000
    )

    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_resized = cv2.resize(mel_db, (224, 224))
    mel_resized = mel_resized[..., np.newaxis]

    mel_resized = (mel_resized - mel_resized.min()) / (
        mel_resized.max() - mel_resized.min()
    )

    mel_resized = np.repeat(mel_resized, 3, axis=-1)

    return mel_resized[np.newaxis, ...].astype(np.float32)


# ---------- Run Inference ----------
def run_inference():
    input_data = preprocess_audio(TEMP_AUDIO)

    prediction = model.predict(input_data, verbose=0)[0]

    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index])

    print("\n🪱 ===== Inference Result =====")
    print(f"Predicted Stage : {CLASSES[predicted_index]}")
    print(f"Confidence      : {confidence:.4f}")
    print("==============================\n")


# ---------- Main ----------
if __name__ == "__main__":
    record_audio()
    run_inference()
