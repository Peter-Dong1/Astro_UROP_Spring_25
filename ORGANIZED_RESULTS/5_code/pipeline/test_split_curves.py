import pickle
import os

total = 0
for i in range(12):
    with open(f"split_light_curves/light_curves_partition_{i:02d}.pkl", "rb") as f:
        curves = pickle.load(f)
        total += len(curves)
print(f"Total light curves across all chunks: {total}")
