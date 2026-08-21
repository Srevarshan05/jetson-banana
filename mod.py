import tkinter as tk
import threading
import time
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

TAMIL_LABELS = {
    'stage_1': 'இலை புழு முதல் கட்டத்தில் உள்ளது',
    'stage_2': 'இலை புழு இரண்டாவது கட்டத்தில் உள்ளது',
    'stage_3': 'இலை புழு மூன்றாவது கட்டத்தில் உள்ளது',
    'stage_4': 'இலை புழு நான்காவது கட்டத்தில் உள்ளது',
    'stage_5': 'இலை புழு ஐந்தாவது கட்டத்தில் உள்ளது',
    'no_larvae': 'இலை புழு இல்லை'
}


RECORD_DURATION = 10
SAMPLE_RATE = 44100
# ==========================================


# -------- Load Model --------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully")


# -------- Preprocess --------
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


# -------- Main Scan Process --------
def run_scan():
    status_label.config(text="🎙 Recording... 10s")
    result_label.config(text="")
    root.update()

    # Record audio
    recording = sd.rec(int(RECORD_DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='int16')

    for i in range(RECORD_DURATION, 0, -1):
        status_label.config(text=f"🎙 Recording... {i}s")
        root.update()
        time.sleep(1)

    sd.wait()
    sf.write(TEMP_AUDIO, recording, SAMPLE_RATE)

    # Loading animation
    for _ in range(3):
        status_label.config(text="🔄 Analyzing.")
        root.update()
        time.sleep(0.3)
        status_label.config(text="🔄 Analyzing..")
        root.update()
        time.sleep(0.3)
        status_label.config(text="🔄 Analyzing...")
        root.update()
        time.sleep(0.3)

    # Inference
    input_data = preprocess_audio(TEMP_AUDIO)
    prediction = model.predict(input_data, verbose=0)[0]

    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index])

    english_label = CLASSES[predicted_index]
    tamil_label = TAMIL_LABELS[english_label]

    status_label.config(text="✅ Scan Complete")

    result_label.config(
        text=f"Prediction: {english_label}\n"
             f"தமிழ்: {tamil_label}\n"
             f"Confidence: {confidence:.2f}"
    )


def start_scan():
    thread = threading.Thread(target=run_scan)
    thread.start()


# -------- GUI --------
root = tk.Tk()
root.title("Banana Weevil Detection")
root.geometry("420x320")
root.configure(bg="#f4f4f4")

title_label = tk.Label(root,
                       text="Banana Weevil Detection System",
                       font=("Arial", 16, "bold"),
                       bg="#f4f4f4")
title_label.pack(pady=15)

scan_button = tk.Button(root,
                        text="Start Scan",
                        command=start_scan,
                        font=("Arial", 14),
                        bg="#4CAF50",
                        fg="white",
                        height=2,
                        width=15)
scan_button.pack(pady=20)

status_label = tk.Label(root,
                        text="Ready",
                        font=("Arial", 12),
                        bg="#f4f4f4")
status_label.pack(pady=10)

result_label = tk.Label(root,
                        text="",
                        font=("Noto Sans Tamil", 14),
                        fg="blue",
                        bg="#f4f4f4")
result_label.pack(pady=15)

root.mainloop()
