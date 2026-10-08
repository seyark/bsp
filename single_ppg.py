# Single PPG Pulse Extraction

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

data = np.loadtxt("ppg.csv", delimiter=",")
raw = data

# High pass filter, 0.05 Hz cut-off (original sampling rate 500 Hz)
b, a = butter(1, 0.05, btype='high', fs=500)
data = filtfilt(b, a, data)

# Downsample 500 Hz -> 100 Hz
s = data[0:len(data):5]
fs = 100
dlen = len(s)
t = np.arange(dlen) / fs

# Peak detection (positive turning point above 50% threshold)
threshold = 0.5 * max(s)
mod_data = np.where(s >= threshold, s, 0)

peak_pos = []
min_gap = int(0.3 * fs)          # two pulses can't be closer than 0.3 s
for i in range(1, dlen - 1):
    if mod_data[i] > mod_data[i - 1] and mod_data[i] >= mod_data[i + 1]:
        if len(peak_pos) > 0 and (i - peak_pos[-1]) < min_gap:
            if s[i] > s[peak_pos[-1]]:
                peak_pos[-1] = i
        else:
            peak_pos.append(i)
peak_pos = np.array(peak_pos)
print("Peak positions: ", peak_pos)

# Onset detection (negative turning point, searching back from each peak)
onset_pos = []
for P in peak_pos:
    o = -1
    for j in range(P - 1, 0, -1):
        if s[j] < s[j - 1] and s[j] <= s[j + 1]:
            o = j
            break
    if o != -1:                  # skip peaks with no onset found
        onset_pos.append(o)
onset_pos = np.array(onset_pos)
print("Onset positions: ", onset_pos)

# Extract one pulse: from the kth onset to the next onset
k = 1                            # 2nd pulse (the 1st may be cut by the start)
start = onset_pos[k]
stop = onset_pos[k + 1]
pulse = s[start:stop + 1]
t_pulse = np.arange(len(pulse)) / fs

# Pulse parameters
pk = np.argmax(pulse)            # systolic peak within the pulse
amp = pulse[pk] - pulse[0]
rise = pk / fs
dur = (stop - start) / fs
area = np.sum(pulse - pulse[0]) / fs

print("\nExtracted pulse number: ", k + 1)
print("Start sample: ", start, "   Stop sample: ", stop)
print("Pulse duration (s): ", round(dur, 3))
print("Systolic amplitude: ", round(amp, 4))
print("Rise time (s): ", round(rise, 3))
print("Pulse area: ", round(area, 4))

# Plot full signal with the selected pulse highlighted
plt.figure(figsize=(12, 4))
plt.title("PPG Signal with Selected Pulse")
plt.plot(t, s, label="PPG")
plt.plot(t[peak_pos], s[peak_pos], "r*", label="Peaks")
plt.plot(t[onset_pos], s[onset_pos], "gv", label="Onsets")
plt.axvspan(start / fs, stop / fs, color="yellow", alpha=0.4, label="Extracted pulse")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.show()

# Plot the single pulse
plt.figure(figsize=(8, 4))
plt.title("Single PPG Pulse")
plt.plot(t_pulse, pulse, "b", linewidth=2)
plt.plot(t_pulse[pk], pulse[pk], "r*", markersize=12, label="Systolic peak")
plt.plot(t_pulse[0], pulse[0], "gv", markersize=10, label="Onset")
plt.plot(t_pulse[-1], pulse[-1], "gv", markersize=10)
plt.fill_between(t_pulse, pulse, pulse[0], color="orange", alpha=0.3, label="Pulse area")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
