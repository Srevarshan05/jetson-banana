import tkinter as tk
import threading
import time
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tensorflow as tf


# ================= CONFIG =================

MODEL_PATH = "/home/sre/Desktop/Banana/jetson-banana/weevil_classifier_full_precision.tflite"
TEMP_AUDIO = "/home/sre/Desktop/Banana/jetson-banana/temp_recording.wav"

CLASSES = [
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
    "stage_5",
    "no_larvae"
]

TAMIL_LABELS = {
    "stage_1": "இலை புழு முதல் கட்டத்தில் உள்ளது",
    "stage_2": "இலை புழு இரண்டாவது கட்டத்தில் உள்ளது",
    "stage_3": "இலை புழு மூன்றாவது கட்டத்தில் உள்ளது",
    "stage_4": "இலை புழு நான்காவது கட்டத்தில் உள்ளது",
    "stage_5": "இலை புழு ஐந்தாவது கட்டத்தில் உள்ளது",
    "no_larvae": "இலை புழு இல்லை"
}

RECORD_DURATION = 10
SAMPLE_RATE = 44100

# ==========================================


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite Model Loaded Successfully")

print(
    "Input shape:",
    input_details[0]["shape"]
)

print(
    "Input type:",
    input_details[0]["dtype"]
)

print(
    "Output shape:",
    output_details[0]["shape"]
)

print(
    "Output type:",
    output_details[0]["dtype"]
)


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def preprocess_audio(
    file_path,
    sr=16000,
    duration=5,
    n_mels=128,
    target_shape=(224, 224)
):

    # --------------------------------------------
    # Load audio
    # --------------------------------------------

    audio, _ = librosa.load(
        file_path,
        sr=sr,
        duration=duration
    )

    # --------------------------------------------
    # Generate Mel Spectrogram
    # --------------------------------------------

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmax=2000
    )

    # --------------------------------------------
    # Convert to dB
    # --------------------------------------------

    mel_spec_db = librosa.power_to_db(
        mel_spec,
        ref=np.max
    )

    # --------------------------------------------
    # Resize to 224 x 224
    # --------------------------------------------

    mel_spec_db = tf.image.resize(
        mel_spec_db[..., np.newaxis],
        target_shape
    ).numpy()

    # --------------------------------------------
    # Normalize
    # --------------------------------------------

    min_value = mel_spec_db.min()
    max_value = mel_spec_db.max()

    if max_value != min_value:

        mel_spec_db = (
            mel_spec_db - min_value
        ) / (
            max_value - min_value
        )

    else:

        mel_spec_db = np.zeros_like(
            mel_spec_db,
            dtype=np.float32
        )

    # --------------------------------------------
    # Convert grayscale → RGB
    # --------------------------------------------

    mel_spec_rgb = np.repeat(
        mel_spec_db,
        3,
        axis=-1
    )

    return mel_spec_rgb.astype(
        np.float32
    )


# ============================================================
# TFLITE INFERENCE
# ============================================================

def predict_audio(file_path):

    # --------------------------------------------
    # Preprocess
    # --------------------------------------------

    input_data = preprocess_audio(
        file_path
    )

    # Add batch dimension
    input_data = input_data[
        np.newaxis,
        ...
    ]

    print(
        "Inference input shape:",
        input_data.shape
    )

    # --------------------------------------------
    # Handle input datatype
    # --------------------------------------------

    expected_dtype = input_details[0]["dtype"]

    if input_data.dtype != expected_dtype:

        input_data = input_data.astype(
            expected_dtype
        )

    # --------------------------------------------
    # Run TFLite inference
    # --------------------------------------------

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interpreter.invoke()

    # --------------------------------------------
    # Get prediction
    # --------------------------------------------

    prediction = interpreter.get_tensor(
        output_details[0]["index"]
    )

    prediction = prediction[0]

    predicted_index = int(
        np.argmax(prediction)
    )

    confidence = float(
        prediction[predicted_index]
    )

    predicted_class = CLASSES[
        predicted_index
    ]

    return (
        predicted_class,
        confidence,
        prediction
    )


# ============================================================
# MAIN SCAN PROCESS
# ============================================================

def run_scan():

    try:

        # --------------------------------------------
        # Recording status
        # --------------------------------------------

        status_label.config(
            text="🎙 Recording... 10s"
        )

        result_label.config(
            text=""
        )

        root.update()

        # --------------------------------------------
        # Record audio
        # --------------------------------------------

        recording = sd.rec(
            int(
                RECORD_DURATION *
                SAMPLE_RATE
            ),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16"
        )

        for i in range(
            RECORD_DURATION,
            0,
            -1
        ):

            status_label.config(
                text=f"🎙 Recording... {i}s"
            )

            root.update()

            time.sleep(1)

        # Wait for recording
        sd.wait()

        # --------------------------------------------
        # Save recording
        # --------------------------------------------

        sf.write(
            TEMP_AUDIO,
            recording,
            SAMPLE_RATE
        )

        print(
            "Audio saved:",
            TEMP_AUDIO
        )

        # --------------------------------------------
        # Analyzing animation
        # --------------------------------------------

        for _ in range(3):

            status_label.config(
                text="🔄 Analyzing."
            )

            root.update()

            time.sleep(0.3)

            status_label.config(
                text="🔄 Analyzing.."
            )

            root.update()

            time.sleep(0.3)

            status_label.config(
                text="🔄 Analyzing..."
            )

            root.update()

            time.sleep(0.3)

        # --------------------------------------------
        # TFLite prediction
        # --------------------------------------------

        predicted_class, confidence, prediction = (
            predict_audio(TEMP_AUDIO)
        )

        tamil_label = TAMIL_LABELS[
            predicted_class
        ]

        # --------------------------------------------
        # Scan complete
        # --------------------------------------------

        status_label.config(
            text="✅ Scan Complete"
        )

        result_label.config(
            text=(
                f"Prediction: {predicted_class}\n"
                f"தமிழ்: {tamil_label}\n"
                f"Confidence: {confidence:.2f}"
            )
        )

        print("\n==============================")
        print("PREDICTION")
        print("==============================")
        print(
            "Class:",
            predicted_class
        )
        print(
            "Confidence:",
            confidence
        )
        print(
            "Raw prediction:",
            prediction
        )
        print("==============================\n")

    except Exception as e:

        status_label.config(
            text="❌ Error"
        )

        result_label.config(
            text=(
                f"{type(e).__name__}: {e}"
            )
        )

        print(
            "\nERROR DURING SCAN:"
        )

        print(e)


# ============================================================
# START SCAN
# ============================================================

def start_scan():

    thread = threading.Thread(
        target=run_scan,
        daemon=True
    )

    thread.start()


# ============================================================
# GUI
# ============================================================

root = tk.Tk()

root.title(
    "Banana Weevil Detection"
)

root.geometry(
    "420x320"
)

root.configure(
    bg="#f4f4f4"
)


# ============================================================
# TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="Banana Weevil Detection System",
    font=("Arial", 16, "bold"),
    bg="#f4f4f4"
)

title_label.pack(
    pady=15
)


# ============================================================
# SCAN BUTTON
# ============================================================

scan_button = tk.Button(
    root,
    text="Start Scan",
    command=start_scan,
    font=("Arial", 14),
    bg="#4CAF50",
    fg="white",
    height=2,
    width=15
)

scan_button.pack(
    pady=20
)


# ============================================================
# STATUS
# ============================================================

status_label = tk.Label(
    root,
    text="Ready",
    font=("Arial", 12),
    bg="#f4f4f4"
)

status_label.pack(
    pady=10
)


# ============================================================
# RESULT
# ============================================================

result_label = tk.Label(
    root,
    text="",
    font=("Noto Sans Tamil", 14),
    fg="blue",
    bg="#f4f4f4"
)

result_label.pack(
    pady=15
)


# ============================================================
# START GUI
# ============================================================

root.mainloop()