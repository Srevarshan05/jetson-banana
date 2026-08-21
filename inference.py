import tensorflow as tf
import numpy as np
import librosa

def load_and_preprocess_audio(file_path, sr=16000, duration=5, n_mels=128, target_shape=(224, 224)):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, fmax=2000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = tf.image.resize(mel_spec_db[..., np.newaxis], target_shape).numpy()
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    return np.repeat(mel_spec_db, 3, axis=-1)

# Load and configure the TFLite model
interpreter = tf.lite.Interpreter(model_path='/home/sre/Desktop/Banana/jetson-banana/weevil_classifier_full_precision.tflite')  # Adjust path
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
classes = ['stage_1', 'stage_2', 'stage_3', 'stage_4', 'stage_5', 'no_larvae']

# Load and preprocess the test file
audio_path = '/home/sre/Desktop/Banana/jetson-banana/test data/stage3 (mp3cut.net) (1).wav'  # Adjust path to your test file
input_data = load_and_preprocess_audio(audio_path)[np.newaxis, ...]

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
prediction = interpreter.get_tensor(output_details[0]['index'])
predicted_class = classes[np.argmax(prediction)]
print(f"Predicted class (TFLite): {predicted_class}")
