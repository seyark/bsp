# ECG Analysis - HR, QRS complex, P wave and T wave

import numpy as np
import matplotlib.pyplot as plt

fs = 125
data = np.loadtxt('C:/Users/mec/Downloads/EB5-6/ECG normal.csv', delimiter=',')
dlen = len(data)

# ================= R PEAK DETECTION =================
Threshold = 0.5 * max(data)
print("Threshold is ", Threshold)

mod_data = []
for i in data:
    if i < Threshold:
        mod_data.append(0)
    else:
        mod_data.append(i)

rpeakvalue = []
rpeakpos = []
for i in range(1, dlen - 1):
    if mod_data[i] > mod_data[i - 1] and mod_data[i] > mod_data[i + 1]:
        rpeakvalue.append(mod_data[i])
        rpeakpos.append(i)

print("R peak values = ", np.array(rpeakvalue))
print("R peak positions = ", rpeakpos)

# ================= HEART RATE =================
rvalue = []
for i in range(len(rpeakpos) - 1):
    rvalue.append(rpeakpos[i + 1] - rpeakpos[i])

print("RR intervals = ", rvalue)
mean_rr = np.mean(rvalue)
print("mean RR = ", mean_rr)
bpm = round((60 / mean_rr) * fs)
print("Heart Rate = ", bpm, "bpm")

# ================= Q AND S DETECTION =================
q = []
s = []
for R in rpeakpos:
    # Q = local minimum within 5 samples BEFORE R
    for j in range(R - 5, R + 1):
        if data[j] < data[j + 1] and data[j] < data[j - 1]:
            q.append(j)
            break
    # S = local minimum within 5 samples AFTER R
    for j in range(R, R + 6):
        if data[j] < data[j + 1] and data[j] < data[j - 1]:
            s.append(j)
            break

q = np.array(q)
s = np.array(s)
print("\nQ positions = ", q)
print("S positions = ", s)

# ================= QRS PARAMETERS =================
qrs_amp = []
for i in range(len(rpeakpos)):
    qrs_amp.append(data[rpeakpos[i]] - min(data[q[i]], data[s[i]]))
print("QRS amplitude of each beat = ", np.array(qrs_amp))
print("Average QRS amplitude = ", round(np.mean(qrs_amp), 2))

qrs = s - q
width = np.mean(qrs) / fs
print("QRS width of each beat (samples) = ", qrs)
print("normal qrs width is 0.07-0.10 s")
print("QRS width = ", round(width, 4), "s")

# ================= P WAVE (referenced to Q) =================
p_pos = []
p_amp = []
for Q in q:
    if Q - 20 < 0:
        continue
    d = data[Q - 20:Q]
    pos = np.argmax(d)
    p_pos.append(Q - 20 + pos)
    p_amp.append(max(d))

p_pos = np.array(p_pos)
p_amp = np.array(p_amp)
print("\nP wave positions = ", p_pos)
print("P wave amplitudes = ", p_amp)
print("Average P amplitude = ", round(np.mean(p_amp), 2))

# P onset and terminal = where the signal returns to baseline
p_onset = []
p_term = []
for P in p_pos:
    for k in range(P, P - 9, -1):
        if data[k] <= 0:
            p_onset.append(k)
            break
    for k in range(P, P + 9):
        if data[k] <= 0:
            p_term.append(k)
            break

p_onset = np.array(p_onset)
p_term = np.array(p_term)
pwave = p_term - p_onset
print("P onset = ", p_onset)
print("P terminal = ", p_term)
print("normal duration of p wave is 0.08-0.11 s")
print("duration of p wave = ", round(np.mean(pwave) / fs, 4), "s")

# ================= T WAVE (referenced to S) =================
t_pos = []
t_amp = []
for S in s:
    if S + 30 >= dlen:
        continue
    r = data[S:S + 30]
    c = np.argmax(r)
    t_pos.append(S + c)
    t_amp.append(max(r))

t_pos = np.array(t_pos)
t_amp = np.array(t_amp)
print("\nT wave positions = ", t_pos)
print("T wave amplitudes = ", t_amp)
print("Average T amplitude = ", round(np.mean(t_amp), 2))

t_onset = []
t_term = []
for T in t_pos:
    for k in range(T, T - 21, -1):
        if data[k] <= 0:
            t_onset.append(k)
            break
    for k in range(T, T + 21):
        if data[k] <= 0:
            t_term.append(k)
            break

t_onset = np.array(t_onset)
t_term = np.array(t_term)
twave = t_term - t_onset
print("T onset = ", t_onset)
print("T terminal = ", t_term)
print("normal duration of t wave is 0.10-0.25 s")
print("duration of t wave = ", round(np.mean(twave) / fs, 4), "s")

# ================= PLOTS =================
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Original ECG")
plt.plot(data)

plt.subplot(3, 1, 2)
plt.title("Thresholded signal")
plt.plot(mod_data)

plt.subplot(3, 1, 3)
plt.title("Detected P, Q, R, S, T")
plt.plot(data)
plt.plot(rpeakpos, data[rpeakpos], "r*", label="R")
plt.plot(q, data[q], "gv", label="Q")
plt.plot(s, data[s], "bv", label="S")
plt.plot(p_pos, data[p_pos], "mo", label="P")
plt.plot(t_pos, data[t_pos], "co", label="T")
plt.legend()
plt.xlabel("Sample number")

plt.tight_layout()
plt.show()
