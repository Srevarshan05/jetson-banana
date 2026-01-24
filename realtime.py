import tensorflow as tf
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import os
from datetime import datetime

def load_and_preprocess_audio(file_path, sr=16000, duration=5, n_mels=128, target_shape=(224, 224)):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=2000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = tf.image.resize(mel_spec_db[..., np.newaxis], target_shape).numpy()
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    return np.repeat(mel_spec_db, 3, axis=-1)

# === Step 1: Record audio from USB mic (30 seconds)
record_duration = 30  # seconds
fs = 44100
channels = 1
device_index = 3  # Your USB mic index
temp_file = "/tmp/temp_recording.wav"

print("🎙️ Recording from mic for 30 seconds...")
recording = sd.rec(int(record_duration * fs), samplerate=fs, channels=channels, dtype='int16', device=device_index)
sd.wait()
sf.write(temp_file, recording, fs)
print("✅ Recording complete.")

# === Step 2: Load and configure the TFLite model
interpreter = tf.lite.Interpreter(model_path='/home/admin/Desktop/banana/weevil_classifier_full_precision.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
classes = ['stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5', 'no_larvae']

# === Step 3: Load and preprocess the recorded file
input_data = load_and_preprocess_audio(temp_file)[np.newaxis, ...]

# === Step 4: Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
predicted_class = classes[np.argmax(prediction)]
print(f"🪱 Predicted class (TFLite): {predicted_class}")
