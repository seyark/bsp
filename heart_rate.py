# Heart Rate from ECG

import numpy as np
import matplotlib.pyplot as plt
fs = 125
data = np.loadtxt("ECG normal.csv")
plt.title("Original Data")
plt.plot(data)
plt.show()

length = len(data)
print("length of data is: ", length)

max_value = max(data)
print("max value is: ", max_value)

th = max_value/2
print("Threshold is: ", th)

new_data = []
for i in data:
    if i >= th:
        new_data = np.append(new_data, i)
    else:
        new_data = np.append(new_data, 0)

# print("new data is: ", new_data)
plt.title("New Data")
plt.plot(new_data)
plt.show()

r_peak = []
r_pos = []

for i in range(len(new_data)):
    if new_data[i] > new_data[i-1] and new_data[i] > new_data[i+1]:
        r_peak = np.append(r_peak, new_data[i])
        r_pos = np.append(r_pos, i)

print("R peak values: ", r_peak)
print("R peak positions: ", r_pos)
plt.title("R peak vs R pos")
plt.plot(data)
plt.plot(r_pos, r_peak, "r" "*")
plt.show()

rr_diff = np.diff(r_pos)
print("Time interval between adj. R-R: ", r_diff)
rr_mean = np.mean(rr_diff)
print("Average RR interval: ", rr_mean)

hr = round((60/rr_mean)*fs)
print("Heart Rate: ", hr)
