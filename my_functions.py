import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def draw_landmarks(image, results):
    """
    Draw hand landmarks on the image with styled connections.

    Args:
        image (numpy.ndarray): The input BGR image.
        results: The landmark results from MediaPipe Hands.
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )


def image_process(image, model):
    """
    Process a BGR image through MediaPipe Hands.

    Args:
        image (numpy.ndarray): The input BGR image.
        model: The MediaPipe Hands object.

    Returns:
        results: Processed results containing hand landmarks.
    """
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    image.flags.writeable = True
    return results


def keypoint_extraction(results):
    """
    Extract and normalize hand keypoints from MediaPipe results.
    Uses only the dominant (first detected) hand.
    Normalizes landmarks relative to wrist position for invariance.

    Args:
        results: The processed results from MediaPipe Hands.

    Returns:
        keypoints (numpy.ndarray): Flattened array of 63 values (21 landmarks x 3).
                                   Returns zeros if no hand detected.
    """
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]  # Use first detected hand

        # Extract raw landmarks
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])

        # Normalize relative to wrist (landmark 0)
        wrist = landmarks[0]
        landmarks -= wrist

        # Scale normalization — divide by max distance from wrist
        scale = np.max(np.abs(landmarks))
        if scale > 0:
            landmarks /= scale

        return landmarks.flatten()
    else:
        return np.zeros(63)


def get_hand_bbox(image, results, padding=20):
    """
    Get bounding box around the detected hand.

    Args:
        image (numpy.ndarray): Input image.
        results: MediaPipe Hands results.
        padding (int): Extra pixels around the hand.

    Returns:
        tuple: (x, y, w, h) of bounding box, or None if no hand.
    """
    if not results.multi_hand_landmarks:
        return None

    h, w = image.shape[:2]
    hand = results.multi_hand_landmarks[0]

    x_coords = [lm.x * w for lm in hand.landmark]
    y_coords = [lm.y * h for lm in hand.landmark]

    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_max = min(h, int(max(y_coords)) + padding)

    return (x_min, y_min, x_max - x_min, y_max - y_min)


def draw_ui_overlay(image, current_letter, word, sentence, confidence, stable_counter, stable_threshold):
    """
    Draw the complete UI overlay on the camera frame.

    Args:
        image: Camera frame.
        current_letter (str): Currently predicted letter.
        word (str): Current word being formed.
        sentence (list): List of completed words.
        confidence (float): Prediction confidence (0-1).
        stable_counter (int): Frames the current letter has been stable.
        stable_threshold (int): Frames needed before letter is accepted.
    """
    h, w = image.shape[:2]

    # ── Semi-transparent top banner ──
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    # ── Title ──
    cv2.putText(image, "ISL Alphabet Recognizer",
                (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 220, 255), 1, cv2.LINE_AA)

    # ── Sentence display ──
    full_sentence = ' '.join(sentence) + (' ' + word if word else '')
    cv2.putText(image, full_sentence if full_sentence.strip() else "Start signing...",
                (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Bottom panel ──
    overlay2 = image.copy()
    cv2.rectangle(overlay2, (0, h - 90), (w, h), (15, 15, 25), -1)
    cv2.addWeighted(overlay2, 0.80, image, 0.20, 0, image)

    # ── Big letter display ──
    if current_letter:
        cv2.putText(image, current_letter,
                    (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 2.8, (100, 230, 100), 3, cv2.LINE_AA)

    # ── Confidence bar ──
    bar_x, bar_y, bar_w, bar_h = 120, h - 72, 180, 14
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill = int(bar_w * confidence)
    bar_color = (80, 200, 80) if confidence > 0.85 else (80, 160, 230) if confidence > 0.6 else (80, 80, 200)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(image, f"Confidence: {confidence:.0%}",
                (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # ── Stability progress dots ──
    dot_label_x = 120
    cv2.putText(image, "Stability:",
                (dot_label_x, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    for i in range(stable_threshold):
        cx = dot_label_x + 75 + i * 18
        cy = h - 48
        color = (80, 230, 80) if i < stable_counter else (60, 60, 60)
        cv2.circle(image, (cx, cy), 6, color, -1)

    # ── Current word ──
    cv2.putText(image, f"Word: {word}",
                (120, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA)

    # ── Controls hint ──
    hints = "[SPACE] Add space  [BKSP] Delete  [ESC] Quit"
    cv2.putText(image, hints,
                (w - 420, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)