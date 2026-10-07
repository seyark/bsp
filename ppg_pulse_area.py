import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

data = np.loadtxt("ppg.csv", delimiter=",")
raw = data

b, a = butter(1, 0.005, btype='high', analog=False)
y = filtfilt(b, a, data)
data = y

down_data = data[0:len(data):5]
s = down_data

threshold = 0.5 * max(s)
fs = 100

mod_data = []
for i in (s):
    if i < threshold:
        mod_data = np.append(mod_data, 0)
    else:
        mod_data = np.append(mod_data, i)

dlen = len(s)

peak_value = []
peak_pos = []
for i in range(1, dlen - 1):
    if ((mod_data[i] > mod_data[i - 1])) and ((mod_data[i] > mod_data[i + 1])):
        peak_value = np.append(peak_value, mod_data[i])
        peak_pos = np.append(peak_pos, i)

peak_pos = peak_pos.astype(int)
print("peak value", peak_value)
print("peak position ", peak_pos)

pp_diff = []
for i in range(len(peak_pos) - 1):
    pp_diff.append(peak_pos[i + 1] - peak_pos[i])

pp_mean = np.mean(pp_diff)
print("Peak to peak intervals (samples): ", np.array(pp_diff))
print("Average interval (seconds): ", round(pp_mean / fs, 4))

hr = round((60 / pp_mean) * fs)
print("Heart Rate: ", hr, "bpm")

# ==================================================================
# ONSET IDENTIFICATION - NEGATIVE TURNING POINT ALGORITHM
# ==================================================================
# Walk backwards from each systolic peak until a sample is found that
# is lower than both its neighbours. That is the negative turning
# point (the foot of the pulse) where the slope changes from
# negative to positive.

onset_pos = []
onset_value = []

for P in peak_pos:
    o = -1
    for j in range(P - 1, 0, -1):
        if (s[j] < s[j - 1]) and (s[j] < s[j + 1]):
            o = j
            break
    onset_pos.append(o)
    onset_value.append(s[o])

onset_pos = np.array(onset_pos)
onset_value = np.array(onset_value)

print("\n--- Onset identification (negative turning point) ---")
print("Onset positions (samples): ", onset_pos)

# ---------- Time index ----------
onset_time = onset_pos / fs
peak_time = peak_pos / fs

print("Onset time index (seconds): ", np.round(onset_time, 3))
print("Peak time index (seconds): ", np.round(peak_time, 3))

# ---------- Systolic amplitude ----------
sys_amp = []
rise_time = []
for k in range(len(peak_pos)):
    sys_amp.append(s[peak_pos[k]] - s[onset_pos[k]])
    rise_time.append((peak_pos[k] - onset_pos[k]) / fs)

sys_amp = np.array(sys_amp)
rise_time = np.array(rise_time)

print("\n--- Systolic amplitude ---")
print("Systolic amplitude of each pulse: ", np.round(sys_amp, 4))
print("MEAN SYSTOLIC AMPLITUDE = ", round(np.mean(sys_amp), 4))
print("Rise time of each pulse (s): ", np.round(rise_time, 3))
print("Mean rise time (s) = ", round(np.mean(rise_time), 4))

# ---------- Pulse area by summation ----------
# Area = sum of all samples from one onset to the next onset,
# measured above the onset level, multiplied by the sample interval.
#          area = SUM( x[n] - x[onset] ) * (1/fs)

pulse_area = []
pulse_dur = []

for k in range(len(onset_pos) - 1):
    start = onset_pos[k]
    stop = onset_pos[k + 1]

    total = 0
    for n in range(start, stop):
        total = total + (s[n] - s[start])

    area = total * (1 / fs)
    pulse_area.append(area)
    pulse_dur.append((stop - start) / fs)

pulse_area = np.array(pulse_area)
pulse_dur = np.array(pulse_dur)

print("\n--- Pulse area (discrete integration) ---")
print("Number of complete pulses: ", len(pulse_area))
print("Pulse duration of each pulse (s): ", np.round(pulse_dur, 3))
print("Pulse area of each pulse: ", np.round(pulse_area, 4))
print("MEAN PULSE AREA = ", round(np.mean(pulse_area), 4))

# ==================================================================
# PLOTS
# ==================================================================
t = np.arange(dlen) / fs

plt.figure(figsize=(12, 5))
plt.title("Down sampled PPG with systolic peaks and onsets")
plt.plot(t, s, label="PPG")
plt.plot(peak_time, s[peak_pos], "r*", markersize=11, label="Systolic peak")
plt.plot(onset_time, s[onset_pos], "gv", markersize=9, label="Onset (foot)")
plt.axhline(threshold, color="k", linestyle="--", linewidth=0.8, label="Threshold")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Single pulse showing amplitude and shaded area
k = 5
st = onset_pos[k]
en = onset_pos[k + 1]

plt.figure(figsize=(10, 6))
plt.title("Pulse %d : systolic amplitude and pulse area" % (k + 1))
plt.plot(t[st:en + 1], s[st:en + 1], "b-", linewidth=1.8)
plt.fill_between(t[st:en], s[st:en], s[st], alpha=0.25, color="skyblue",
                 label="Pulse area = %.4f" % pulse_area[k])

plt.plot(t[peak_pos[k]], s[peak_pos[k]], "r*", markersize=14, label="Systolic peak")
plt.plot(t[st], s[st], "gv", markersize=11, label="Onset")

plt.annotate("", xy=(t[peak_pos[k]], s[st]),
             xytext=(t[peak_pos[k]], s[peak_pos[k]]),
             arrowprops=dict(arrowstyle="<->", color="red"))
plt.text(t[peak_pos[k]] + 0.03, (s[st] + s[peak_pos[k]]) / 2,
         "Systolic\namplitude\n%.4f" % sys_amp[k], color="red", fontsize=10)

plt.axhline(s[st], color="green", linestyle="--", linewidth=0.9)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.show()

# Beat wise summary
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.title("Systolic amplitude per pulse")
plt.plot(sys_amp, "o-", color="crimson")
plt.axhline(np.mean(sys_amp), color="k", linestyle="--",
            label="mean = %.4f" % np.mean(sys_amp))
plt.xlabel("Pulse number")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.title("Pulse area per pulse")
plt.plot(pulse_area, "o-", color="teal")
plt.axhline(np.mean(pulse_area), color="k", linestyle="--",
            label="mean = %.4f" % np.mean(pulse_area))
plt.xlabel("Pulse number")
plt.ylabel("Area")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
