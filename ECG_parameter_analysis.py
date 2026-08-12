# ECG Parameter Analysis - QRS, P wave, T wave

import numpy as np
import matplotlib.pyplot as plt

fs = 125
data = np.loadtxt("ECG normal.csv")

plt.title("Original Data")
plt.plot(data)
plt.show()

# Step 1: R peak detection (same as HR experiment)
max_value = max(data)
th = max_value / 2
print("Threshold is: ", th)

new_data = []
for i in data:
    if i >= th:
        new_data = np.append(new_data, i)
    else:
        new_data = np.append(new_data, 0)

r_peak = []
r_pos = []
for i in range(1, len(new_data) - 1):
    if new_data[i] > new_data[i - 1] and new_data[i] > new_data[i + 1]:
        r_peak = np.append(r_peak, new_data[i])
        r_pos = np.append(r_pos, i)

r_pos = r_pos.astype(int)
print("R peak values: ", r_peak)
print("R peak positions: ", r_pos)

# Step 2: Q and S using +/- 5 samples from R
q_pos = []
q_peak = []
s_pos = []
s_peak = []
p_pos = []
p_peak = []
t_pos = []
t_peak = []
qrs_amp = []
qrs_width = []

for R in r_pos:
    # skip beats too close to the edges of the record
    if R - 30 < 0 or R + 50 >= len(data):
        continue

    # Q = minimum in the 5 samples BEFORE R
    q = np.argmin(data[R - 5:R]) + (R - 5)

    # S = minimum in the 5 samples AFTER R
    s = np.argmin(data[R:R + 6]) + R

    # P = maximum between 30 and 6 samples before R (atrial depolarisation)
    p = np.argmax(data[R - 30:R - 6]) + (R - 30)

    # T = maximum between 10 and 50 samples after R (ventricular repolarisation)
    t = np.argmax(data[R + 10:R + 50]) + (R + 10)

    q_pos = np.append(q_pos, q)
    q_peak = np.append(q_peak, data[q])
    s_pos = np.append(s_pos, s)
    s_peak = np.append(s_peak, data[s])
    p_pos = np.append(p_pos, p)
    p_peak = np.append(p_peak, data[p])
    t_pos = np.append(t_pos, t)
    t_peak = np.append(t_peak, data[t])

    qrs_amp = np.append(qrs_amp, data[R] - min(data[q], data[s]))
    qrs_width = np.append(qrs_width, (s - q) / fs)

# Step 3: Print results
print("\n--- QRS Complex ---")
print("Q positions: ", q_pos)
print("S positions: ", s_pos)
print("QRS amplitude of each beat: ", qrs_amp)
print("QRS width of each beat (s): ", qrs_width)
print("Average QRS amplitude: ", round(np.mean(qrs_amp), 2))
print("Average QRS width (s): ", round(np.mean(qrs_width), 4))

print("\n--- P wave ---")
print("P wave positions: ", p_pos)
print("P wave amplitudes: ", p_peak)
print("Average P amplitude: ", round(np.mean(p_peak), 2))

print("\n--- T wave ---")
print("T wave positions: ", t_pos)
print("T wave amplitudes: ", t_peak)
print("Average T amplitude: ", round(np.mean(t_peak), 2))

# Step 4: Plot all detected points
plt.figure(figsize=(12, 5))
plt.title("ECG with P, Q, R, S, T marked")
plt.plot(data, label="ECG")
plt.plot(r_pos, data[r_pos], "r*", label="R")
plt.plot(q_pos.astype(int), q_peak, "gv", label="Q")
plt.plot(s_pos.astype(int), s_peak, "bv", label="S")
plt.plot(p_pos.astype(int), p_peak, "mo", label="P")
plt.plot(t_pos.astype(int), t_peak, "co", label="T")
plt.xlabel("Sample number")
plt.ylabel("Amplitude")
plt.legend()
plt.show()
