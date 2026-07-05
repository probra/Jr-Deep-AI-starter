# ============================================================
#  ROCK  PAPER  SCISSORS  -  beginner version
#  Run this file, make a shape with your hand, and play!
# ============================================================


# ------------------------------------------------------------
#  SETUP - you do NOT need to understand this part.
#  It just turns on the camera and loads your trained AI model.
#  Leave it exactly as it is and scroll down to "THE GAME".
# ------------------------------------------------------------
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import random
import time
import cv2
import numpy as np
from tf_keras.models import load_model
from tf_keras.layers import DepthwiseConv2D


# This little block fixes an error when loading a Teachable Machine model.
class PatchedDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)


# Load your trained model and the list of label names.
model = load_model(
    "converted_keras/keras_model_c33.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": PatchedDepthwiseConv2D},
)
class_names = open("converted_keras/labels.txt", "r").readlines()

# Turn the camera on, count down, and take ONE photo.
camera = cv2.VideoCapture(0)

print("Get your hand ready!")
# Keep grabbing frames for ~3 seconds so the camera has time to wake up
# and adjust its brightness. The LAST frame is the photo we keep.
countdown_end = time.time() + 3
image = None
while time.time() < countdown_end:
    ret, image = camera.read()
camera.release()
print("Snap! Photo taken.")

# Mirror the photo left-to-right, because the Teachable Machine website
# shows a mirrored ("selfie") preview - so the AI was trained on mirrored
# images. This makes our photo match what it learned from.
image = cv2.flip(image, 1)

# The camera photo is wide, but the AI was trained on SQUARE pictures.
# So we cut out the middle square first (this is what the website does).
height, width = image.shape[:2]
side = min(height, width)
top = (height - side) // 2
left = (width - side) // 2
image = image[top:top + side, left:left + side]

# Shrink the square photo and swap colours to RGB (the AI expects RGB,
# but the camera gives BGR), then hand it to the AI to get a guess.
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
image = (image / 127.5) - 1
prediction = model.predict(image, verbose=0)
guess_number = np.argmax(prediction)

# Turn the AI's guess into a simple word: "rock", "paper" or "scissors".
player = class_names[guess_number]   # looks like "0 Rock"
player = player[2:]                  # remove the "0 " -> "Rock"
player = player.strip()              # remove spaces/newlines -> "Rock"
player = player.lower()              # make it lowercase -> "rock"
# ------------------------------------------------------------
#  END OF SETUP
# ------------------------------------------------------------


# ============================================================
#  THE GAME  -  this is the part you can read and change!
# ============================================================

# 1) Pick a random move for the computer.



# 2) Show what each player chose.


# 3) Work out who won, using only if statements.

# If both chose the same thing, it's a draw.
