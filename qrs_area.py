# QRS Complex Area from ECG

import numpy as np
import matplotlib.pyplot as plt

fs = 125
data = np.loadtxt("ECG Normal.csv", delimiter=",")
plt.title("Original Data")
plt.plot(data)
plt.show()

length = len(data)
print("length of data is: ", length)

baseline = np.median(data)
data = data - baseline
print("baseline is: ", baseline)

max_value = max(data)
print("max value is: ", max_value)

th = max_value / 2
print("Threshold is: ", th)

new_data = []
for i in data:
    if i >= th:
        new_data = np.append(new_data, i)
    else:
        new_data = np.append(new_data, 0)

plt.title("New Data")
plt.plot(new_data)
plt.show()

r_peak = []
r_pos = []
for i in range(1, len(new_data) - 1):
    if new_data[i] > new_data[i - 1] and new_data[i] >= new_data[i + 1]:
        r_peak = np.append(r_peak, new_data[i])
        r_pos = np.append(r_pos, i)

r_pos = r_pos.astype(int)
print("R peak values: ", r_peak)
print("R peak positions: ", r_pos)

q_pos = []
s_pos = []
for R in r_pos:
    if R - 5 >= 0 and R + 6 <= length:
        q = R - 5 + np.argmin(data[R - 5:R])
        s = R + 1 + np.argmin(data[R + 1:R + 6])
        q_pos = np.append(q_pos, q)
        s_pos = np.append(s_pos, s)

q_pos = q_pos.astype(int)
s_pos = s_pos.astype(int)
print("Q positions: ", q_pos)
print("S positions: ", s_pos)

plt.title("Q, R, S points")
plt.plot(data)
plt.plot(r_pos, data[r_pos], "r*")
plt.plot(q_pos, data[q_pos], "g*")
plt.plot(s_pos, data[s_pos], "b*")
plt.show()

qrs_area = []
for i in range(len(q_pos)):
    total = 0
    for n in range(q_pos[i], s_pos[i] + 1):
        total = total + abs(data[n])
    area = total * (1 / fs)
    qrs_area = np.append(qrs_area, area)


qrs_area = []
for i in range(len(q_pos)):
    area = np.sum(np.abs(data[q_pos[i]:s_pos[i] + 1])) * (1 / fs)
    qrs_area = np.append(qrs_area, area)
  

print("QRS area of each beat: ", qrs_area)

mean_area = np.mean(qrs_area)
print("Mean QRS area: ", mean_area)

qrs_width = (s_pos - q_pos) / fs
print("Mean QRS width in seconds: ", np.mean(qrs_width))

plt.title("QRS area of each beat")
plt.stem(qrs_area)
plt.show()
