"""
model_train.py
──────────────
Trains a CNN model on the ISL keypoint data prepared by dataset_prep.py.

Run AFTER dataset_prep.py:
    python model_train.py

Saves:
    my_model.keras   ← trained model (used by main.py)
    label_map.npy    ← maps model output index → letter (used by main.py)
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════
DATA_PATH   = "data"
MODEL_PATH  = "my_model.keras"
LABEL_PATH  = "label_map.npy"
EPOCHS      = 80
BATCH_SIZE  = 32
TEST_SPLIT  = 0.15
# ══════════════════════════════════════


def load_data(data_path, actions):
    landmarks, labels = [], []
    label_map = {label: idx for idx, label in enumerate(actions)}

    for action in actions:
        action_path = os.path.join(data_path, action)
        files = sorted([f for f in os.listdir(action_path) if f.endswith('.npy')])

        for f in files:
            kp = np.load(os.path.join(action_path, f))
            landmarks.append(kp)
            labels.append(label_map[action])

    return np.array(landmarks), np.array(labels), actions


def build_model(input_dim, num_classes):
    """
    Dense neural network on flattened 63-dim keypoint vectors.
    Works better than CNN for 1D landmark data.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train():
    # ── Validate data folder ──
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"❌ '{DATA_PATH}' not found. Run dataset_prep.py first."
        )

    actions = sorted([
        d for d in os.listdir(DATA_PATH)
        if os.path.isdir(os.path.join(DATA_PATH, d))
    ])

    if len(actions) == 0:
        raise ValueError("❌ No action folders found in data/.")

    print(f"✅ Classes ({len(actions)}): {actions}")

    # ── Load ──
    X, Y, actions = load_data(DATA_PATH, actions)
    print(f"   Dataset shape: {X.shape}  |  Labels: {Y.shape}")

    # ── Save label map ──
    np.save(LABEL_PATH, np.array(actions))
    print(f"   Label map saved → {LABEL_PATH}")

    # ── Split ──
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=TEST_SPLIT,
        random_state=42,
        stratify=Y
    )
    print(f"   Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # ── Build ──
    model = build_model(X.shape[1], len(actions))
    model.summary()

    # ── Callbacks ──
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        )
    ]

    # ── Train ──
    print("\n🚀 Training...\n")
    model.fit(
        X_train, Y_train,
        validation_data=(X_test, Y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # ── Save ──
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")

    # ── Evaluate ──
    preds = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = metrics.accuracy_score(Y_test, preds)
    print(f"✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")

    # ── Per-class report ──
    print(metrics.classification_report(Y_test, preds, target_names=actions))


if __name__ == "__main__":
    train()