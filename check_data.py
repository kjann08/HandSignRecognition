import numpy as np

data = np.load("A.npy", allow_pickle=True)
print("Shape:", data.shape)
print("First sample:", data[0] if len(data) > 0 else "Empty")
