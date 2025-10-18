import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = "hand_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

label = input("Enter label (e.g., A, B, Hello): ")
samples = []

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Flatten landmarks
                data = []
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

                # ✅ Save only when "s" is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    samples.append(data)
                    print(f"Saved sample {len(samples)}")

        cv2.putText(frame, f"Samples: {len(samples)}  Press 's' to save, 'q' to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save collected samples
samples = np.array(samples)
np.save(os.path.join(DATA_DIR, f"{label}.npy"), samples)
print(f"✅ Saved {len(samples)} samples for {label}")
