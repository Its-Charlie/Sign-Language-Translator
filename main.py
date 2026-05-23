"""
main.py
───────
Real-time ISL Alphabet Recognizer.

Controls:
    SPACE     → Add space (finalize current word, start new one)
    BACKSPACE → Delete last letter from current word
    ESC       → Quit

Run:
    python main.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import keyboard

from my_functions import (
    image_process,
    draw_landmarks,
    keypoint_extraction,
    get_hand_bbox,
    draw_ui_overlay
)

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════
MODEL_PATH      = "my_model.keras"
LABEL_PATH      = "label_map.npy"
CONFIDENCE_THRESHOLD = 0.80   # Min confidence to accept a prediction
STABLE_THRESHOLD     = 12     # Frames a letter must be stable before accepted
COOLDOWN_FRAMES      = 20     # Frames to wait after a letter is added
# ══════════════════════════════════════


def load_model_and_labels():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"❌ Model '{MODEL_PATH}' not found. Run model_train.py first."
        )
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(
            f"❌ Label map '{LABEL_PATH}' not found. Run model_train.py first."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    actions = np.load(LABEL_PATH, allow_pickle=True)
    print(f"✅ Model loaded. Classes: {list(actions)}")
    return model, actions


def run():
    model, actions = load_model_and_labels()

    # ── State ──
    word            = ""        # Current word being built
    sentence        = []        # Completed words
    current_letter  = ""        # Last accepted letter
    stable_letter   = ""        # Letter being tracked for stability
    stable_counter  = 0         # How many frames stable_letter has held
    cooldown        = 0         # Frames remaining in cooldown
    confidence      = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot access camera.")

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    print("\n🤟 ISL Alphabet Recognizer running.")
    print("   SPACE → add space | BACKSPACE → delete | ESC → quit\n")

    with mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # Mirror for natural feel

            # ── Process frame ──
            results = image_process(frame, hands)
            draw_landmarks(frame, results)

            # ── Draw hand bounding box ──
            bbox = get_hand_bbox(frame, results, padding=25)
            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (80, 200, 80), 2)

            # ── Predict ──
            if cooldown > 0:
                cooldown -= 1
            else:
                keypoints = keypoint_extraction(results)

                if not np.all(keypoints == 0):
                    pred = model.predict(keypoints[np.newaxis, :], verbose=0)[0]
                    confidence = float(np.max(pred))
                    predicted_letter = actions[np.argmax(pred)]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        # ── Stability check ──
                        if predicted_letter == stable_letter:
                            stable_counter += 1
                        else:
                            stable_letter  = predicted_letter
                            stable_counter = 1

                        # ── Accept letter after stable_threshold frames ──
                        if stable_counter >= STABLE_THRESHOLD:
                            if predicted_letter != current_letter:
                                word           += predicted_letter
                                current_letter  = predicted_letter
                                stable_counter  = 0
                                cooldown        = COOLDOWN_FRAMES
                    else:
                        # Low confidence — reset stability
                        stable_letter  = ""
                        stable_counter = 0
                        confidence     = float(np.max(pred))  # still show bar
                else:
                    # No hand → reset everything
                    stable_letter  = ""
                    stable_counter = 0
                    confidence     = 0.0

            # ── Keyboard controls ──
            if keyboard.is_pressed('space'):
                if word:
                    sentence.append(word)
                    word           = ""
                    current_letter = ""
                    stable_counter = 0
                    cooldown       = COOLDOWN_FRAMES

            if keyboard.is_pressed('backspace'):
                if word:
                    word           = word[:-1]
                    current_letter = word[-1] if word else ""
                    cooldown       = COOLDOWN_FRAMES

            # ── Draw UI ──
            draw_ui_overlay(
                frame,
                current_letter,
                word,
                sentence,
                confidence,
                stable_counter,
                STABLE_THRESHOLD
            )

            cv2.imshow("ISL Alphabet Recognizer", frame)

            # ── Exit on ESC or window close ──
            if cv2.waitKey(1) & 0xFF == 27:
                break
            if cv2.getWindowProperty("ISL Alphabet Recognizer", cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Application closed.")


if __name__ == "__main__":
    run()