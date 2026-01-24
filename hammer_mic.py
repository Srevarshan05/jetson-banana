import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
from pathlib import Path

HAMMER_FILE = "/tmp/hammer.wav"
MODEL_PATH = "/home/admin/Desktop/banana/weevil_classifier_full_precision.tflite"
CLASS_NAMES = ['stage_1','stage_2','stage_3','stage_4','stage_5','no_larvae']

SR = 16000            # preprocessing resample rate used earlier
WINDOW_SECONDS = 5.0  # model input window length
HOP_SECONDS = 2.5     # overlap between windows (set = WINDOW_SECONDS for non-overlap)
N_MELS = 128
FMAX = 2000
TARGET_SHAPE = (224, 224)
CONFIDENCE_THRESHOLD = 0.50  # if max prob < threshold -> label no_larvae

def load_windowed_audio(path, sr=SR, window=WINDOW_SECONDS, hop=HOP_SECONDS):
    audio, orig_sr = librosa.load(path, sr=sr, mono=True)
    total_len = audio.shape[0]
    win_len = int(window * sr)
    hop_len = int(hop * sr)
    starts = list(range(0, max(1, total_len - 1), hop_len))
    windows = []
    for s in starts:
        e = s + win_len
        chunk = audio[s:e]
        if len(chunk) < win_len:
            chunk = np.pad(chunk, (0, win_len - len(chunk)))
        windows.append((s, chunk))
    return windows, sr

def preprocess_chunk(chunk, sr=SR, n_mels=N_MELS, fmax=FMAX, target_shape=TARGET_SHAPE):
    mel = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    img = mel_norm[..., np.newaxis].astype('float32')
    img_resized = tf.image.resize(img, target_shape).numpy()
    img_3 = np.repeat(img_resized, 3, axis=-1)
    return img_3[np.newaxis, ...].astype('float32')

# Load interpreter
interp = tf.lite.Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
input_details = interp.get_input_details()[0]
output_details = interp.get_output_details()[0]

# Helper to set input respecting quantization
def set_input(interp, input_array):
    dtype = input_details['dtype']
    if np.issubdtype(dtype, np.floating):
        tensor = input_array.astype(dtype)
    else:
        # quantized model: get scale/zero_point
        scale, zp = input_details['quantization']
        if scale == 0:
            # fallback
            tensor = (input_array * 255.0).astype(dtype)
        else:
            tensor = np.round(input_array / scale + zp).astype(dtype)
    interp.set_tensor(input_details['index'], tensor)

def run_window_inference(file_path):
    windows, sr = load_windowed_audio(file_path)
    results = []
    for start, chunk in windows:
        inp = preprocess_chunk(chunk, sr=sr)
        set_input(interp, inp)
        interp.invoke()
        out = interp.get_tensor(output_details['index'])
        # dequantize if needed
        out_dtype = output_details['dtype']
        scale, zp = output_details.get('quantization', (0.0, 0))
        if np.issubdtype(out_dtype, np.integer) and scale != 0:
            out = (out.astype('float32') - zp) * scale
        probs = out.squeeze()
        pred_idx = int(np.argmax(probs))
        pred_name = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else f"class_{pred_idx}"
        results.append((start, pred_name, probs))
        # print per-window details
        start_time = start / sr
        print(f"win start {start_time:.2f}s -> pred: {pred_name}  probs: {np.round(probs,4)}")
    return results

if __name__ == "__main__":
    p = Path(HAMMER_FILE)
    if not p.exists():
        print("Hammer file not found:", HAMMER_FILE)
        raise SystemExit(1)
    res = run_window_inference(HAMMER_FILE)

    # Aggregate results
    votes = {}
    avg_probs = np.zeros(len(CLASS_NAMES), dtype=float)
    for _, _, probs in res:
        avg_probs += probs
        pred = CLASS_NAMES[int(np.argmax(probs))]
        votes[pred] = votes.get(pred, 0) + 1
    avg_probs /= max(1, len(res))
    # Decide final label: if highest avg prob < threshold -> no_larvae
    final_idx = int(np.argmax(avg_probs))
    if avg_probs[final_idx] < CONFIDENCE_THRESHOLD:
        final_label = "no_larvae"
    else:
        final_label = CLASS_NAMES[final_idx]

    print("\n=== SUMMARY ===")
    print("Window count:", len(res))
    print("Votes:", votes)
    print("Average probs:", np.round(avg_probs,4))
    print("Final label (with threshold {:.2f}): {}".format(CONFIDENCE_THRESHOLD, final_label))
