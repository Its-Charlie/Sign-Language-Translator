"""
dataset_prep.py
───────────────
Converts the raw ISL image dataset (from Kaggle) into MediaPipe hand keypoints
saved as .npy files — ready for training.

Expected input folder structure (set DATASET_PATH below):
    ISL_Dataset/
        A/
            img1.jpg
            img2.jpg
            ...
        B/
            ...
        ...
        Z/
            ...

Output structure (auto-created):
    data/
        A/
            0.npy
            1.npy
            ...
        B/
            ...

Usage:
    1. Download the ISL dataset from Kaggle:
       https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl
    2. Extract it and set DATASET_PATH to the folder containing A-Z subfolders.
    3. Run:  python dataset_prep.py
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from my_functions import image_process, keypoint_extraction

# ══════════════════════════════════════════════
# CONFIG — change DATASET_PATH to your folder
# ══════════════════════════════════════════════
DATASET_PATH = "ISL_Dataset"   # Folder with A/, B/, ..., Z/ sub-folders
OUTPUT_PATH  = "data"          # Where keypoint .npy files will be saved
MAX_IMAGES   = 300             # Max images per class to process (set None for all)
# ══════════════════════════════════════════════

def prepare_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"❌ Dataset folder '{DATASET_PATH}' not found.\n"
            f"   Download from: https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl\n"
            f"   Then set DATASET_PATH to the extracted folder path."
        )

    actions = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    if len(actions) == 0:
        raise ValueError(f"❌ No class subfolders found inside '{DATASET_PATH}'.")

    print(f"✅ Found {len(actions)} classes: {actions}\n")

    with mp.solutions.hands.Hands(
        static_image_mode=True,         # Static mode for images (not video)
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        total_saved   = 0
        total_skipped = 0

        for action in actions:
            action_dir = os.path.join(DATASET_PATH, action)
            out_dir    = os.path.join(OUTPUT_PATH, action)
            os.makedirs(out_dir, exist_ok=True)

            image_files = [
                f for f in os.listdir(action_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
            ]

            if MAX_IMAGES:
                image_files = image_files[:MAX_IMAGES]

            saved   = 0
            skipped = 0

            for img_file in image_files:
                img_path = os.path.join(action_dir, img_file)
                image = cv2.imread(img_path)

                if image is None:
                    skipped += 1
                    continue

                results = image_process(image, hands)
                keypoints = keypoint_extraction(results)

                # Skip if no hand was detected (all zeros)
                if np.all(keypoints == 0):
                    skipped += 1
                    continue

                npy_path = os.path.join(out_dir, str(saved))
                np.save(npy_path, keypoints)
                saved += 1

            print(f"  [{action}]  Saved: {saved}  |  Skipped (no hand): {skipped}")
            total_saved   += saved
            total_skipped += skipped

    print(f"\n✅ Done! Total saved: {total_saved} | Total skipped: {total_skipped}")
    print(f"   Keypoints stored in: '{OUTPUT_PATH}/'")


if __name__ == "__main__":
    prepare_dataset()