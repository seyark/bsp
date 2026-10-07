import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

data = np.loadtxt("ppg.csv", delimiter=",")
#cropping data to 2000 points
data = data[0:2000]

# High-pass filter
b, a = butter(4, 0.2, btype='high', analog=False, fs=500)
y = filtfilt(b, a, data)

# Low-pass filter
b, a = butter(4, 200, btype='low', analog=False, fs=500)
yy = filtfilt(b, a, y)

plt.subplot(6, 1, 1)
plt.plot(data, color="Blue")

plt.subplot(6, 1, 2)
plt.plot(yy, color="Red")


#2nd order derivative of filtered signal
yyy = []


for i in range(2, len(yy)-2):
    p = (2 * yy[i - 2]) - (yy[i - 1]) - (2 * yy[i]) - (yy[i + 1]) + (2 * yy[i + 2])
    yyy=np.append(yyy,p)

plt.subplot(6, 1, 3)
plt.plot(yyy, color="Green")
plt.show()

#smoothening output

s=[]
for i in range(31,len(yyy)):
    a=0
    for k in range(1,33):
        a=(((yyy[i-k+1])**2)*(32-k+1))
        
    s=np.append(s,a)
    
b, a = butter(4, 10, btype='low', analog=False, fs=500)
yy = filtfilt(b, a, s)
s=yy       
plt.subplot(6, 1, 4)
plt.plot(s, color="Orange")
plt.show()
       
threshold = 0.50 * max(s)
print("Threshold is: " , threshold)  

mod_data = []
for i in s:
    if i < threshold:
        mod_data = np.append(mod_data, 0)
    else:
        mod_data = np.append(mod_data, i)
        
        
plt.subplot(6, 1, 5)
plt.plot(mod_data, color="pink")
plt.show()        

peak_value=[]
peak_pos=[]
for i in range(1, len(s) - 1):
    if ((mod_data[i] > mod_data[i - 1])) and ((mod_data[i] > mod_data[i + 1])):
        peak_value = np.append(peak_value, mod_data[i])
        peak_pos= np.append(peak_pos, i)


print("peak value", peak_value)  
print("peak posistion ", peak_pos)
dn=[]

for i in range (len(peak_pos)-1):
    d=peak_pos[i+1]-peak_pos[i]
    
    if d>100:
        dn=np.append(dn,peak_pos[i])
        
print(dn)

d_n = []

for i in dn:
    l = int( i - 50)
    u = int( i + 50)

    k = np.argmin(s[l:u])
    ppg_pos = l + k

    d_n = np.append(d_n, ppg_pos)

print("dichrotic notch positions are :", d_n)

plt.subplot(6, 1, 6)
plt.plot(y, color="blue")
plt.scatter(d_n, y[d_n.astype(int)], color="red")
plt.show()
