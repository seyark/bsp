# Experiment 5: Linear convolution of two sequences x(n) and h(n)

# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Enter the two sequences (numbers separated by spaces, e.g. 1 2 3 4)
x = np.array(list(map(float, input("Enter the sequence x(n): ").split())))
h = np.array(list(map(float, input("Enter the sequence h(n): ").split())))

# Step 3: Length of the sequences
n1 = len(x)
n2 = len(h)
print("Length of x(n), n1 =", n1)
print("Length of h(n), n2 =", n2)

# Step 4: Length of the output sequence
n = n1 + n2 - 1
print("Length of output sequence, n = n1 + n2 - 1 =", n)

# Step 5: Zero pad both sequences to length n
xp = np.append(x, np.zeros(n - n1))
hp = np.append(h, np.zeros(n - n2))
print("Zero padded x(n):", xp)
print("Zero padded h(n):", hp)

# Step 6: Convolution using the general formula
# y(i) = sum over k = 0 to i of x(k) * h(i - k)
y = np.zeros(n)
for i in range(n):
    for k in range(i + 1):
        y[i] = y[i] + xp[k] * hp[i - k]

print("Convolution output y(n):", y)
print("Verification using np.convolve:", np.convolve(x, h))

# Step 7: Plot the two sequences and the convolution output
plt.figure(figsize=(8, 8))

plt.subplot(3, 1, 1)
plt.stem(np.arange(n1), x)
plt.title("Input Sequence x(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.stem(np.arange(n2), h)
plt.title("Impulse Response h(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 3)
plt.stem(np.arange(n), y)
plt.title("Convolution Output y(n) = x(n) * h(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Step 8: Stop
