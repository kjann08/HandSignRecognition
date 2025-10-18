# train_final.py
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import random
import math

DATA_DIR = "hand_data"   # your folder with A.npy, B.npy, ..., SPACE.npy, BACKSPACE.npy
MODEL_PATH = "model.pkl"
AUGMENT_FACTOR = 3       # how many augmented copies to make per sample (set to 0 to disable)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------- MediaPipe topology (for visualize if needed) ----------
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ---------- Normalization ----------
def normalize_landmarks(landmarks):
    # landmarks: 63-long flat list or array
    lm = np.array(landmarks).reshape(-1, 3).astype(np.float32)
    # translate so wrist (index 0) is origin
    lm -= lm[0]
    # optionally remove z (if noisy) - keep z for now
    # scale by max distance
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 0:
        lm /= max_dist
    return lm.flatten()

# ---------- Augmentation helpers ----------
def rotate_xy(lm_flat, angle_deg):
    lm = np.array(lm_flat).reshape(-1, 3)
    angle = math.radians(angle_deg)
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s],[s, c]])
    xy = lm[:, :2] @ R.T
    lm[:, :2] = xy
    return lm.flatten()

def jitter(lm_flat, sigma=0.01):
    lm = np.array(lm_flat).reshape(-1, 3)
    lm += np.random.normal(scale=sigma, size=lm.shape)
    return lm.flatten()

def scale(lm_flat, factor):
    lm = np.array(lm_flat).reshape(-1, 3)
    lm[:, :2] *= factor
    return lm.flatten()

def augment_sample(sample):
    out = []
    base = sample.copy()
    # small rotations, scales and jitter
    for _ in range(AUGMENT_FACTOR):
        a = random.uniform(-20, 20)  # degrees
        s = random.uniform(0.92, 1.08)
        j = random.uniform(0.0, 0.02)
        s1 = rotate_xy(base, a)
        s2 = scale(s1, s)
        s3 = jitter(s2, sigma=j)
        out.append(s3)
    return out

# ---------- Load data ----------
X, y = [], []
label_idx_to_name = {}
cnt_total = 0

for idx, fname in enumerate(sorted(os.listdir(DATA_DIR))):
    if not fname.lower().endswith(".npy"):
        continue
    label = fname.replace(".npy","")
    path = os.path.join(DATA_DIR, fname)
    raw = np.load(path, allow_pickle=True)
    # ensure shape (n_samples, 63)
    raw = np.array(raw)
    if raw.ndim == 1:
        raw = raw.reshape(-1, 63)  # try to fix single-sample-wrapped
    samples = []
    for s in raw:
        try:
            norm = normalize_landmarks(s)
            samples.append(norm)
        except Exception as e:
            continue
    if len(samples) == 0:
        continue
    label_idx_to_name[idx] = label
    cnt_total += len(samples)
    for s in samples:
        X.append(s)
        y.append(idx)
        # augmentation
        if AUGMENT_FACTOR > 0:
            for aug in augment_sample(s):
                X.append(normalize_landmarks(aug))  # normalize after augment
                y.append(idx)

X = np.array(X)
y = np.array(y)
X, y = shuffle(X, y, random_state=RANDOM_SEED)

print(f"✅ Loaded data: {X.shape} samples, {len(label_idx_to_name)} classes (after augment: {AUGMENT_FACTOR}x)")

# ---------- Train/Test split ----------
if len(X) < 10:
    raise SystemExit("Not enough data to train. Collect more samples per class (>=20 recommended).")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y)

# ---------- Build pipeline ----------
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(hidden_layer_sizes=(256,128), max_iter=800, random_state=RANDOM_SEED))
])

print("⏳ Training model (this may take a minute)...")
clf.fit(X_train, y_train)

# ---------- Evaluate ----------
acc = clf.score(X_test, y_test)
print(f"✅ Model accuracy: {acc:.4f}")

y_pred = clf.predict(X_test)
print("\n📊 Classification Report:")
target_names = [label_idx_to_name[i] for i in sorted(label_idx_to_name.keys())]
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ---------- Save model + labels ----------
with open(MODEL_PATH, "wb") as f:
    pickle.dump((clf, label_idx_to_name), f)

print(f"✅ Saved model to {MODEL_PATH}")
