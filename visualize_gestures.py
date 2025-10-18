# visualize_gestures.py
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "hand_data"
OUT_DIR = "gesture_images"
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def normalize_landmarks(landmarks):
    lm = np.array(landmarks).reshape(-1,3)
    lm -= lm[0]
    maxd = (np.linalg.norm(lm, axis=1)).max()
    if maxd>0:
        lm /= maxd
    return lm

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".npy"):
        continue
    label = fname.replace(".npy","")
    arr = np.load(os.path.join(DATA_DIR, fname), allow_pickle=True)
    if arr.ndim == 1:
        arr = arr.reshape(-1,63)
    norms = []
    for s in arr:
        norms.append(normalize_landmarks(s))
    norms = np.array(norms)
    mean = np.mean(norms, axis=0)  # shape (21,3)
    mean = mean.reshape(-1,3)

    plt.figure(figsize=(4,6))
    # draw connections
    for a,b in HAND_CONNECTIONS:
        x = [mean[a,0], mean[b,0]]
        y = [-mean[a,1], -mean[b,1]]  # invert y for plotting upright
        plt.plot(x,y, linewidth=3)
    # draw points
    plt.scatter(mean[:,0], -mean[:,1], s=40)
    for i,(x,y) in enumerate(zip(mean[:,0], mean[:,1])):
        plt.text(x, -y, str(i), fontsize=8)

    plt.title(label)
    plt.axis('off')
    out = os.path.join(OUT_DIR, f"{label}.png")
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved", out)
