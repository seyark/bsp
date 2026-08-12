import numpy as np

fs = 125
data = np.loadtxt("ECG normal.csv")

th = data.max() / 2
nd = np.where(data >= th, data, 0)

r_pos = np.flatnonzero((nd[1:-1] > nd[:-2]) & (nd[1:-1] > nd[2:])) + 1
r_peak = data[r_pos]

rr = np.diff(r_pos)
hr = round(60 / rr.mean() * fs)

print("R positions:", r_pos)       # [26 127 230 332 435 538 640 743]
print("Mean RR (samples):", rr.mean())
print("Heart Rate:", hr)           # 73
