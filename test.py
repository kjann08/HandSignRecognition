# test_final_corrected_fixed.py
# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')  # ✅ Fix for Windows Unicode printing

import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque, Counter

MODEL_PATH = "model.pkl"
FLIP_FRAME = True   # True if you collected mirrored frames
BUFFER_SIZE = 12
STABILITY_THRESHOLD = 0.7
CONF_THRESHOLD = 0.80
COOLDOWN = 0.8      # seconds after adding a letter
ADD_DELAY_ON_HAND_REMOVED = 0.6  # seconds of hand absence to confirm candidate

# ---------- Load model ----------
with open(MODEL_PATH, "rb") as f:
    pipeline, label_map = pickle.load(f)

# unify label_map to list: idx->label
if isinstance(label_map, dict):
    idx_to_label = [label_map[i] for i in sorted(label_map.keys())]
else:
    idx_to_label = list(label_map)

# ---------- Normalization ----------
def normalize_landmarks(landmarks):
    lm = np.array(landmarks).reshape(-1, 3)
    lm -= lm[0]
    maxd = (np.linalg.norm(lm, axis=1)).max()
    if maxd > 0:
        lm /= maxd
    return lm.flatten()

# ---------- MediaPipe setup ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------- Video capture ----------
cap = cv2.VideoCapture(0)
pred_buffer = deque(maxlen=BUFFER_SIZE)
time_last_add = 0
candidate = None
candidate_time = 0
last_seen_hand = time.time()
word = ""

print("✋ Show gesture. Press 'a' to force-add candidate, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if FLIP_FRAME:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    hand_present = False
    if res.multi_hand_landmarks:
        for h in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
            hand_present = True
            last_seen_hand = time.time()

            lm = []
            for lmpt in h.landmark:
                lm.extend([lmpt.x, lmpt.y, lmpt.z])
            feats = normalize_landmarks(lm).reshape(1, -1)

            # predict
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(feats)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])
            else:
                idx = int(pipeline.predict(feats)[0])
                conf = 1.0
            label = idx_to_label[idx]

            # append only if confident
            if conf >= CONF_THRESHOLD:
                pred_buffer.append((label, conf))
            else:
                pred_buffer.append(("__LOW__", conf))

            # show live prediction
            cv2.putText(frame, f"{label} {conf:.2f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # no hand detected
        pred_buffer.append((None, 0.0))

    # ---------- Analyze buffer ----------
    if len(pred_buffer) == BUFFER_SIZE:
        labels = [p[0] for p in pred_buffer if p[0] not in (None, "__LOW__")]
        if len(labels) > 0:
            most, cnt = Counter(labels).most_common(1)[0]
            stability = cnt / BUFFER_SIZE
        else:
            most, stability = None, 0.0

        # if stable candidate appears
        if most is not None and stability >= STABILITY_THRESHOLD:
            if candidate != most:
                candidate = most
                candidate_time = time.time()
            # display candidate
            display_label = candidate
            if candidate == "SPACE":
                display_label = "␣"
            elif candidate == "BACKSPACE":
                display_label = "⌫"
            cv2.putText(frame, f"Candidate: {display_label} ({stability*100:.0f}%)",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
        else:
            # if hand removed for enough time, accept candidate
            if candidate is not None and (time.time() - last_seen_hand) > ADD_DELAY_ON_HAND_REMOVED and (time.time() - time_last_add) > COOLDOWN:
                if candidate == "SPACE":
                    word += " "
                elif candidate == "BACKSPACE":
                    word = word[:-1]
                else:
                    word += candidate
                print("Added:", candidate)
                time_last_add = time.time()
                candidate = None
                pred_buffer.clear()
            elif not hand_present:
                pass
            else:
                candidate = None

    # ---------- Draw current word ----------
    cv2.putText(frame, f"Word: {word}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("HandSign Word Builder", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        # force-add current candidate
        if candidate is not None and (time.time() - time_last_add) > COOLDOWN:
            if candidate == "SPACE":
                word += " "
            elif candidate == "BACKSPACE":
                word = word[:-1]
            else:
                word += candidate
            print("Force-added:", candidate)
            time_last_add = time.time()
            candidate = None
            pred_buffer.clear()

cap.release()
cv2.destroyAllWindows()
print("Final word:", word)
