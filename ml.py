import RPi.GPIO as GPIO
import time
import numpy as np
from sklearn.linear_model import LogisticRegression

# === Setup GPIO pins ===
RED_LED = 17
GREEN_LED = 27

GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED, GPIO.OUT)
GPIO.setup(GREEN_LED, GPIO.OUT)

# === Train a simple model ===
# Let's say input features are just 2D vectors
X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
y_train = np.array([0, 1, 1, 0])  # 0 = Negative, 1 = Positive

model = LogisticRegression()
model.fit(X_train, y_train)

# === Predict an example ===
# Example test data (you can replace with real input later)
X_test = np.array([[0,0]])
prediction = model.predict(X_test)[0]

# === Show result via LED ===
def blink_led(pin):
    GPIO.output(pin, GPIO.HIGH)
    time.sleep(1)
    GPIO.output(pin, GPIO.LOW)

try:
    if prediction == 1:
        print("Positive class detected. Blinking GREEN LED.")
        blink_led(GREEN_LED)
    else:
        print("Negative class detected. Blinking RED LED.")
        blink_led(RED_LED)
finally:
    GPIO.cleanup()
