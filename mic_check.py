import sounddevice as sd
import soundfile as sf
import numpy as np

OUTPUT_FILE = "boya_test.wav"
DURATION = 5
SAMPLE_RATE = 44100


print("=" * 60)
print("AUDIO DEVICES")
print("=" * 60)

print(sd.query_devices())

print("\n" + "=" * 60)
print("DEFAULT DEVICE")
print("=" * 60)

print(sd.default.device)


# ---------------------------------------------------------
# Find default input device
# ---------------------------------------------------------

input_device = sd.default.device[0]

device_info = sd.query_devices(input_device)

print("\nSelected microphone:")
print(device_info)


# ---------------------------------------------------------
# Record
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RECORDING")
print("=" * 60)

print("Speak into your BOYA M1...")
print(f"Recording for {DURATION} seconds...\n")

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

sd.wait()

print("Recording finished.")


# ---------------------------------------------------------
# Analyze audio level
# ---------------------------------------------------------

peak = np.max(np.abs(recording))
rms = np.sqrt(np.mean(recording ** 2))

print("\nAudio level:")
print(f"Peak : {peak:.6f}")
print(f"RMS  : {rms:.6f}")


if peak < 0.001:
    print("\n⚠️ WARNING: Almost no audio detected.")
elif peak < 0.01:
    print("\n⚠️ Audio detected, but the level is very low.")
else:
    print("\n✅ Microphone audio detected!")


# ---------------------------------------------------------
# Save WAV
# ---------------------------------------------------------

sf.write(
    OUTPUT_FILE,
    recording,
    SAMPLE_RATE
)

print(f"\nSaved recording to:")
print(OUTPUT_FILE)


# ---------------------------------------------------------
# Playback
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("PLAYBACK")
print("=" * 60)

print("Playing recorded audio...")

sd.play(
    recording,
    SAMPLE_RATE
)

sd.wait()

print("Playback finished.")