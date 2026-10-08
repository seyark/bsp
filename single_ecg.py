import numpy as np
import matplotlib.pyplot as plt
fs = 125
data = np.loadtxt("ECG normal.csv")
t = np.arange(len(data))/fs
plt.title("Original Data")
plt.plot(t, data)
plt.show()

max_value = np.max(data)
th = 0.5 * max_value
print("Max value: ", max_value)
print("Threshold: ", th)

nd = np.where(data >= th, data, 0)
plt.plot(t, nd)
plt.show()

r_pos = []
r_peak = []
for i in range(1, len(nd)-1):
    if nd[i] > nd[i-1] and nd[i] > nd[i+1]:
        r_peak.append(nd[i])
        r_pos.append(i)
r_pos = np.array(r_pos)
r_peak = np.array(r_peak)
print("R peak positions: ", r_pos)
plt.plot(t, nd)
plt.plot(t[r_pos], nd[r_pos], "r*")
plt.axvspan(start / fs, end / fs, color="orange", alpha=0.4)
plt.show()

rr_diff = np.diff(r_pos)
rr_mean = np.mean(rr_diff)
print("Average RR interval: ", rr_mean, "samples")

k = 1

before = int(rr_mean/3)
after = int(2*rr_mean/3)

start = r_pos[k] - before
end = r_pos[k] + after
beat = data[start:end]
t_beat = np.arange(len(beat))/fs
plt.plot(t_beat, beat)
plt.show()

print("beat starts at sample: ", start)
print("beat ends at sample: ", end)
print("beat length: ", len(beat)/fs, "sec")
