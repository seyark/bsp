# Signal Processing from Scratch — Theory, Hand-Worked Examples, and Python

Every topic follows the same shape:

1. **The idea** — what the operation actually does
2. **The formula** — and what each symbol means
3. **Solved numerical example** — worked out by hand, step by step
4. **Loop program** — plain Python, no shortcuts, mirrors the hand calculation
5. **NumPy program** — the same thing done properly
6. **Verification** — the two must agree

The rule throughout: *write the loop version first, check it against your hand calculation, then use NumPy.* If you jump straight to `np.convolve` you will never spot the day you've fed it the wrong thing.

---

# Topic 1 — Discrete-Time Signals

## 1.1 The idea

A discrete-time signal `x[n]` is a list of numbers indexed by an integer `n`. That's it. `n` is not time in seconds — it's the *sample number*. If you sampled at `fs` samples/second, then sample `n` happened at time `t = n/fs` seconds.

The subtlety that trips up every beginner: **a Python list stores only the values, not the index axis.** The list `[1, 2, 3]` could mean a signal living at `n = 0,1,2` or at `n = -5,-4,-3`. You must carry the index axis alongside the values, or you will get convolution start-indices wrong for the rest of your life.

## 1.2 The standard test signals

| Signal | Definition | Meaning |
|---|---|---|
| Unit impulse `δ[n]` | 1 at `n=0`, else 0 | the "1" of convolution |
| Unit step `u[n]` | 1 for `n≥0`, else 0 | switching something on |
| Unit ramp `r[n]` | `n` for `n≥0`, else 0 | `r[n] = n·u[n]` |
| Exponential | `aⁿ·u[n]` | decays if `\|a\|<1`, grows if `\|a\|>1` |
| Sinusoid | `A·sin(2πf₀n/fs + φ)` | the building block of everything |

Note `u[n] - u[n-1] = δ[n]`, and `Σ δ[k]` for `k ≤ n` gives `u[n]`. The step and impulse are each other's difference and running sum.

## 1.3 Solved example

Generate all four signals for `n = -3 … 3`, with `a = 0.5`.

```
n         : -3   -2   -1    0    1     2      3
δ[n]      :  0    0    0    1    0     0      0
u[n]      :  0    0    0    1    1     1      1
r[n]      :  0    0    0    0    1     2      3
0.5ⁿ·u[n] :  0    0    0    1   0.5  0.25  0.125
```

Check `0.5ⁿ` by hand: `0.5⁰=1`, `0.5¹=0.5`, `0.5²=0.25`, `0.5³=0.125`. ✓

## 1.4 Loop program

```python
def gen_signals_loop(start, stop):
    n, delta, step, ramp, expo = [], [], [], [], []
    for i in range(start, stop + 1):
        n.append(i)
        delta.append(1.0 if i == 0 else 0.0)
        step.append(1.0 if i >= 0 else 0.0)
        ramp.append(float(i) if i >= 0 else 0.0)
        expo.append(0.5 ** i if i >= 0 else 0.0)
    return n, delta, step, ramp, expo

n, d, u, r, e = gen_signals_loop(-3, 3)
print("n     :", n)
print("delta :", d)
print("u     :", u)
print("ramp  :", r)
print("expo  :", e)
```

## 1.5 NumPy program

```python
import numpy as np

n = np.arange(-3, 4)                                  # index axis, kept explicitly
delta = (n == 0).astype(float)                        # boolean mask -> 0/1
step  = (n >= 0).astype(float)
ramp  = np.where(n >= 0, n, 0).astype(float)          # where(cond, if_true, if_false)
expo  = np.where(n >= 0, 0.5 ** n.astype(float), 0.0)
```

**What changed:** the `if` inside the loop became a *boolean mask*. `(n == 0)` produces `[False,...,True,...,False]`, and `.astype(float)` turns that into `[0,...,1,...,0]`. `np.where` is the vectorised `if/else`. There is no loop because NumPy applies the comparison to all elements at once.

**Watch out:** `0.5 ** n` with an integer array raises an error for negative `n` (integers can't hold `0.5⁻³` as an int). Cast with `n.astype(float)` first — that's why it's there.

## 1.6 Sampling and aliasing

```python
fs = 100.0                       # samples per second
f0 = 5.0                         # signal frequency in Hz
t  = np.arange(0, 1, 1/fs)       # 100 time points covering 1 second
x  = np.sin(2*np.pi*f0*t)
```

The **Nyquist rule**: you must have `fs > 2·f0`, otherwise higher frequencies fold down and masquerade as lower ones. Concretely, at `fs = 100 Hz` a 95 Hz sine is indistinguishable from a −5 Hz sine:

```python
x5  = np.sin(2*np.pi*5*t)
x95 = np.sin(2*np.pi*95*t)
print(np.max(np.abs(x5 + x95)))   # ~5.6e-14, i.e. they are exact negatives
```

This matters for you directly: ECG needs `fs ≥ 500 Hz` to preserve QRS morphology, EMG needs 1–2 kHz because its power extends to 500 Hz.

---

# Topic 2 — Basic Operations on Signals

## 2.1 The four operations

| Operation | Formula | What happens to values | What happens to index axis |
|---|---|---|---|
| Amplitude scaling | `y[n] = a·x[n]` | multiplied by `a` | unchanged |
| Time shift | `y[n] = x[n−k]` | unchanged | shifts **right** by `k` |
| Time reversal (folding) | `y[n] = x[−n]` | reversed | negated and reversed |
| Time scaling | `y[n] = x[an]` | decimated/expanded | compressed/stretched |

**The shift direction is the single most common mistake.** `x[n−2]` is a *delay* — the signal moves right, to larger `n`. Reason it out: if `x` had its peak at `n=0`, then `y[n]=x[n−2]` peaks when `n−2=0`, i.e. at `n=2`. Later. Delayed.

## 2.2 Solved example — shifting and folding

Let `x[n] = {1, 2, 3, 4}` for `n = 0,1,2,3`.

**Delay by 2:** `y[n] = x[n−2]`
```
values : 1  2  3  4        (unchanged)
n      : 2  3  4  5        (each index + 2)
```

**Fold:** `y[n] = x[−n]`
```
original :  x[0]=1  x[1]=2  x[2]=3  x[3]=4
y[0]=x[0]=1,  y[-1]=x[1]=2,  y[-2]=x[2]=3,  y[-3]=x[3]=4

sorted by n:
n      : -3  -2  -1   0
values :  4   3   2   1
```

## 2.3 Solved example — adding signals on different ranges

`x₁[n] = {1, 2, 3}` on `n = 0,1,2` and `x₂[n] = {1, 1, 1, 1}` on `n = −1,0,1,2`.

You cannot just add the lists — they don't line up. Build a common axis `n = −1 … 2` and pad with zeros:

```
n      :  -1   0   1   2
x₁[n]  :   0   1   2   3
x₂[n]  :   1   1   1   1
sum    :   1   2   3   4
product:   0   1   2   3
```

## 2.4 Loop program

```python
def shift_loop(x, nx, k):
    """y[n] = x[n-k]: values unchanged, index axis moves by +k"""
    return list(x), [i + k for i in nx]

def fold_loop(x, nx):
    """y[n] = x[-n]: reverse values, negate and reverse the index axis"""
    return x[::-1], [-i for i in nx[::-1]]

def scale_loop(x, a):
    out = []
    for v in x:
        out.append(a * v)
    return out

def add_loop(x1, n1, x2, n2):
    lo = min(min(n1), min(n2))
    hi = max(max(n1), max(n2))
    n_out, y = [], []
    for i in range(lo, hi + 1):
        v1 = x1[n1.index(i)] if i in n1 else 0.0   # zero outside the range
        v2 = x2[n2.index(i)] if i in n2 else 0.0
        n_out.append(i)
        y.append(v1 + v2)
    return y, n_out
```

## 2.5 NumPy program

```python
x  = np.array([1., 2., 3., 4.])
nx = np.arange(0, 4)

shifted_vals, shifted_n = x, nx + 2          # delay by 2
folded_vals,  folded_n  = x[::-1], -nx[::-1] # fold
scaled                  = 3 * x

def align_numpy(x1, n1, x2, n2):
    """Put both signals on a common index axis, zero-padded."""
    n = np.arange(min(n1.min(), n2.min()), max(n1.max(), n2.max()) + 1)
    y1 = np.zeros(len(n)); y2 = np.zeros(len(n))
    y1[np.searchsorted(n, n1)] = x1          # place values at the right slots
    y2[np.searchsorted(n, n2)] = x2
    return y1, y2, n

a, b, nc = align_numpy(np.array([1.,2.,3.]), np.arange(0,3),
                       np.array([1.,1.,1.,1.]), np.arange(-1,3))
print(a + b)     # [1. 2. 3. 4.]
print(a * b)     # [0. 1. 2. 3.]
```

**What changed:** the `.index()` lookup inside the loop — which is O(n) per element and genuinely slow — becomes a single `np.searchsorted` call that finds all the insertion slots at once. Scaling and folding become one-liners because NumPy broadcasts the scalar and supports negative-stride slicing.

**If you want a fixed-length shift** (signal stays the same length, zeros fill in):

```python
def shift_fixed(x, k):
    y = np.zeros_like(x)
    if k > 0:   y[k:]  = x[:-k]     # delay
    elif k < 0: y[:k]  = x[-k:]     # advance
    else:       y = x.copy()
    return y
```

---

# Topic 3 — Linear Convolution

This is the central operation. Everything else in DSP is a special case or a consequence.

## 3.1 The idea

If a system is **linear** and **time-invariant** (LTI), then knowing its response to a single impulse `δ[n]` tells you its response to *anything*. That impulse response is called `h[n]`.

Why? Decompose the input into scaled, shifted impulses:

```
x[n] = x[0]·δ[n] + x[1]·δ[n−1] + x[2]·δ[n−2] + …
```

Linearity says the output is the sum of the individual responses. Time-invariance says `δ[n−k]` produces `h[n−k]`. So:

```
y[n] = x[0]·h[n] + x[1]·h[n−1] + x[2]·h[n−2] + …
```

which is exactly

$$y[n] = \sum_{k=-\infty}^{\infty} x[k]\,h[n-k]$$

**Read the formula this way:** *every input sample drops a scaled copy of `h` into the output, starting at that sample's position. Add up all the overlapping copies.* That description is literally the tabular method, and it's the one to hold in your head.

## 3.2 The two mental models

**Model A — flip and slide (matches the formula):**
1. Flip `h[k]` to get `h[−k]`
2. Shift by `n` to get `h[n−k]`
3. Multiply point-by-point with `x[k]`
4. Sum → that gives **one** output sample `y[n]`
5. Repeat for every `n`

**Model B — scale and add (easier to compute by hand):**
For each input sample `x[k]`, write down `x[k]·h[n]` starting at position `k`. Stack them in a staircase and add columns.

They give the same answer. Model A is how the formula reads; Model B is how you'd actually do it on paper.

## 3.3 Length and start index — two rules you must memorise

- **Length:** `len(y) = len(x) + len(h) − 1`
- **Start index:** if `x` starts at `n₁` and `h` starts at `n₂`, then `y` starts at `n₁ + n₂`

NumPy gives you the values but *not* the index axis. You have to apply the second rule yourself. Forgetting it is the number-one source of "my filtered signal is shifted" bugs.

## 3.4 Solved example

`x[n] = {1, 2, 3, 4}` (starting `n=0`), `h[n] = {1, 2, 3}` (starting `n=0`).

Length = 4 + 3 − 1 = **6**. Start index = 0 + 0 = **0**, so `y` lives on `n = 0…5`.

### Tabular method (Model B)

Each row is `x[k]` times the whole of `h`, shifted right by `k`:

```
              n=0   n=1   n=2   n=3   n=4   n=5
x[0]=1 · h :   1     2     3
x[1]=2 · h :         2     4     6
x[2]=3 · h :               3     6     9
x[3]=4 · h :                     4     8    12
─────────────────────────────────────────────────
y[n]       :   1     4    10    16    17    12
```

Column sums: `1`, `2+2=4`, `3+4+3=10`, `6+6+4=16`, `9+8=17`, `12`.

### Same thing from the definition (Model A)

```
y[0] = x[0]h[0]                            = 1·1                 = 1
y[1] = x[0]h[1] + x[1]h[0]                 = 1·2 + 2·1           = 4
y[2] = x[0]h[2] + x[1]h[1] + x[2]h[0]      = 1·3 + 2·2 + 3·1     = 10
y[3] =            x[1]h[2] + x[2]h[1] + x[3]h[0] = 2·3+3·2+4·1   = 16
y[4] =                       x[2]h[2] + x[3]h[1] = 3·3+4·2       = 17
y[5] =                                  x[3]h[2] = 4·3           = 12
```

**Answer:** `y[n] = {1, 4, 10, 16, 17, 12}`, `n = 0…5`

**Sanity check that costs one second:** `Σy` must equal `Σx · Σh`. Here `1+4+10+16+17+12 = 60` and `(1+2+3+4)·(1+2+3) = 10·6 = 60`. ✓ Do this on every convolution you compute by hand.

## 3.5 Loop programs (three ways)

**Method 1 — straight from the definition.** The `if` guard is what stops `h[n−k]` from being indexed out of range.

```python
def conv_loop(x, h):
    N, M = len(x), len(h)
    L = N + M - 1
    y = [0.0] * L
    for n in range(L):                       # for each output sample
        for k in range(N):                   # sum over all input samples
            if 0 <= n - k < M:               # keep h's index legal
                y[n] += x[k] * h[n - k]
    return y
```

**Method 2 — zero-pad first, so no `if` on `h` is needed.** This is closer to the textbook formula where signals are infinite and zero outside their support.

```python
def conv_loop_padded(x, h):
    N, M = len(x), len(h)
    L = N + M - 1
    xp = x + [0.0] * (L - N)                 # pad both to output length
    hp = h + [0.0] * (L - M)
    y = [0.0] * L
    for n in range(L):
        s = 0.0
        for k in range(L):
            if n - k >= 0:
                s += xp[k] * hp[n - k]
        y[n] = s
    return y
```

**Method 3 — tabular / scale-and-add.** This is Model B in code, and it's the cleanest of the three: no index guards at all, because `k + m` can never exceed `L−1`.

```python
def conv_loop_table(x, h):
    L = len(x) + len(h) - 1
    y = [0.0] * L
    for k in range(len(x)):                  # each input sample...
        for m in range(len(h)):              # ...scatters a scaled copy of h
            y[k + m] += x[k] * h[m]
    return y
```

All three print `[1.0, 4.0, 10.0, 16.0, 17.0, 12.0]`.

## 3.6 NumPy programs (three ways)

```python
import numpy as np
x = np.array([1., 2., 3., 4.])
h = np.array([1., 2., 3.])
```

**Method 1 — shift-and-accumulate.** One loop instead of two: the inner loop over `h` becomes a slice assignment.

```python
def conv_numpy_shift(x, h):
    y = np.zeros(len(x) + len(h) - 1)
    for k in range(len(x)):
        y[k:k+len(h)] += x[k] * h            # whole copy of h added at once
    return y
```

**Method 2 — Toeplitz matrix.** Convolution is a linear operation, so it can be written as a matrix multiply `y = H·x`. Each column of `H` is `h` slid down by one.

```python
def conv_numpy_matrix(x, h):
    N, M = len(x), len(h)
    H = np.zeros((N + M - 1, N))
    for k in range(N):
        H[k:k+M, k] = h
    return H @ x
```

For our example, `H` is 6×4:

```
[1 0 0 0]
[2 1 0 0]
[3 2 1 0]
[0 3 2 1]
[0 0 3 2]
[0 0 0 3]
```

Multiply by `x = [1,2,3,4]ᵀ` and you get `[1,4,10,16,17,12]ᵀ`. This view matters later — it's how you'd solve *deconvolution* (given `y` and `h`, recover `x`) as a least-squares problem, which is directly relevant to source separation.

**Method 3 — outer product, sum the anti-diagonals.** The outer product `P[k,m] = x[k]·h[m]` is exactly the tabular method's grid, and `y[n]` is the sum where `k + m = n`.

```python
def conv_numpy_outer(x, h):
    P = np.outer(x, h)                       # P[k,m] = x[k]*h[m]
    L = len(x) + len(h) - 1
    y = np.zeros(L)
    F = np.fliplr(P)                         # anti-diagonals become diagonals
    for n in range(L):
        y[n] = np.trace(F, offset=len(h) - 1 - n)
    return y
```

**And the library function:**

```python
y = np.convolve(x, h)          # [ 1.  4. 10. 16. 17. 12.]
```

## 3.7 The three modes of `np.convolve`

```python
np.convolve(x, h, mode='full')    # [1, 4, 10, 16, 17, 12]  length N+M-1
np.convolve(x, h, mode='same')    # [4, 10, 16, 17]         length max(N,M)
np.convolve(x, h, mode='valid')   # [10, 16]                length |N-M|+1
```

- **`full`** — everything, including the partial-overlap edges. Default.
- **`same`** — centre slice, same length as the longer input. Convenient for filtering, but the first and last `M//2` samples are contaminated by the implicit zero-padding.
- **`valid`** — only the samples where `h` sits entirely inside `x`. No edge artefacts at all, but the signal gets shorter.

For biosignal filtering, `same` is usually what you want, with the understanding that you should discard or ignore the edges.

## 3.8 The impulse is the identity

```python
np.convolve([1,2,3], [1,0,0])    # [1, 2, 3, 0, 0]  -> unchanged
np.convolve([1,2,3], [0,0,1])    # [0, 0, 1, 2, 3]  -> delayed by 2
```

`x * δ[n] = x[n]` and `x * δ[n−k] = x[n−k]`. This is the **sifting property**, and it's the reason `h[n]` — the response to `δ[n]` — characterises the whole system.

---

# Topic 4 — Properties of Convolution

These aren't trivia. Each one corresponds to a physical way of wiring systems together.

| Property | Statement | Physical meaning |
|---|---|---|
| Commutative | `x * h = h * x` | which one you call "signal" and which "system" is arbitrary |
| Associative | `(x * h₁) * h₂ = x * (h₁ * h₂)` | two filters in **cascade** = one filter `h₁ * h₂` |
| Distributive | `x * (h₁ + h₂) = x*h₁ + x*h₂` | two filters in **parallel** = one filter `h₁ + h₂` |
| Identity | `x * δ = x` | the impulse does nothing |
| Shift | `x[n] * δ[n−k] = x[n−k]` | convolving with a shifted impulse delays |
| Sum | `Σy = Σx · Σh` | free correctness check |
| Width | `len(y) = len(x) + len(h) − 1` | filtering always spreads the signal |

## 4.1 Solved example — cascade

`x = {1,2,3}`, `h₁ = {1,−1}`, `h₂ = {2,0,1}`.

Route A: `x * h₁ = {1, 1, 1, −3}`, then `* h₂` →
```
{1,1,1,-3} * {2,0,1} = {2, 2, 3, -5, 1, -3}
```

Route B: `h₁ * h₂ = {1,−1} * {2,0,1} = {2, −2, 1, −1}`, then `x * that` →
```
{1,2,3} * {2,-2,1,-1} = {2, 2, 3, -5, 1, -3}
```

Identical. In practice this means: if you're applying a 50 Hz notch and then a baseline-wander high-pass, you can precompute a single combined kernel and do one convolution instead of two.

## 4.2 Verification program

```python
x  = np.array([1., 2., 3.])
h1 = np.array([1., -1.])
h2 = np.array([2., 0., 1.])

# Commutative
assert np.allclose(np.convolve(x, h1), np.convolve(h1, x))

# Associative
assert np.allclose(np.convolve(np.convolve(x, h1), h2),
                   np.convolve(x, np.convolve(h1, h2)))

# Distributive — must pad h1 to h2's length before adding
h1p = np.pad(h1, (0, len(h2) - len(h1)))
assert np.allclose(np.convolve(x, h1p + h2),
                   np.convolve(x, h1p) + np.convolve(x, h2))

# Sum rule
y = np.convolve(x, h2)
assert np.isclose(y.sum(), x.sum() * h2.sum())
```

Note `np.pad(h1, (0, k))` — the tuple is `(pad_before, pad_after)`. You need this because NumPy won't add arrays of different lengths, whereas the mathematics implicitly treats both as zero outside their support.

---

# Topic 5 — Circular Convolution

## 5.1 The idea

Circular convolution is convolution on a signal that's treated as **periodic** — when a shift runs off the right end, it wraps back around to the left instead of falling into zeros.

You care about it for exactly one reason: **the DFT computes circular convolution, not linear.** Multiply two DFTs and inverse-transform, and what you get back is the circular convolution. If you wanted linear convolution and forgot to zero-pad, your answer will be wrong in a specific, recognisable way (explained in 5.5).

$$y[n] = \sum_{k=0}^{N-1} x[k]\,h[(n-k) \bmod N], \qquad n = 0,1,\dots,N-1$$

Two differences from linear convolution: the `mod N` on the index, and the output length is `N`, not `N+M−1`.

## 5.2 Solved example

`x = {1, 2, 3, 4}`, `h = {1, 0, 2, 1}`, `N = 4`.

`h[(n−k) mod 4]` means: when the index goes negative, add 4.

```
y[0] = x[0]h[0] + x[1]h[-1→3] + x[2]h[-2→2] + x[3]h[-3→1]
     = 1·1 + 2·1 + 3·2 + 4·0 = 1 + 2 + 6 + 0 = 9

y[1] = x[0]h[1] + x[1]h[0] + x[2]h[-1→3] + x[3]h[-2→2]
     = 1·0 + 2·1 + 3·1 + 4·2 = 0 + 2 + 3 + 8 = 13

y[2] = x[0]h[2] + x[1]h[1] + x[2]h[0] + x[3]h[-1→3]
     = 1·2 + 2·0 + 3·1 + 4·1 = 2 + 0 + 3 + 4 = 9

y[3] = x[0]h[3] + x[1]h[2] + x[2]h[1] + x[3]h[0]
     = 1·1 + 2·2 + 3·0 + 4·1 = 1 + 4 + 0 + 4 = 9
```

**Answer:** `y = {9, 13, 9, 9}`

## 5.3 The circulant matrix method

Build `H[i][j] = h[(i−j) mod N]`. Each row is the previous row shifted right by one, with the wrapped element coming around:

```
      j=0  j=1  j=2  j=3
i=0 [  1    1    2    0  ]      <- h[0], h[-1→3], h[-2→2], h[-3→1]
i=1 [  0    1    1    2  ]
i=2 [  2    0    1    1  ]
i=3 [  1    2    0    1  ]
```

Then `y = H·x`:
```
y[0] = 1·1 + 1·2 + 2·3 + 0·4 = 9
y[1] = 0·1 + 1·2 + 1·3 + 2·4 = 13
y[2] = 2·1 + 0·2 + 1·3 + 1·4 = 9
y[3] = 1·1 + 2·2 + 0·3 + 1·4 = 9
```
✓ Same answer.

## 5.4 Loop programs

```python
def circconv_loop(x, h, N=None):
    if N is None:
        N = max(len(x), len(h))
    xp = list(x) + [0.0] * (N - len(x))          # pad to length N if needed
    hp = list(h) + [0.0] * (N - len(h))
    y = [0.0] * N
    for n in range(N):
        for k in range(N):
            y[n] += xp[k] * hp[(n - k) % N]      # the % N is the whole trick
    return y


def circconv_loop_matrix(x, h):
    N = len(x)
    H = [[h[(i - j) % N] for j in range(N)] for i in range(N)]
    y = [0.0] * N
    for i in range(N):
        for j in range(N):
            y[i] += H[i][j] * x[j]
    return y, H
```

Python's `%` returns a non-negative result for a positive modulus, so `(-1) % 4` gives `3` — exactly the wrap you want. (In C you'd need `((n-k) % N + N) % N` because C's `%` can return negative values.)

## 5.5 NumPy programs

```python
def circconv_numpy_roll(x, h):
    """Each x[k] scales a circularly-shifted copy of h."""
    N = len(x)
    y = np.zeros(N)
    for k in range(N):
        y += x[k] * np.roll(h, k)                # np.roll wraps around
    return y


def circconv_numpy_fft(x, h, N=None):
    """The fast way: multiply in the frequency domain."""
    N = N or max(len(x), len(h))
    return np.real(np.fft.ifft(np.fft.fft(x, N) * np.fft.fft(h, N)))
```

Both give `[9, 13, 9, 9]`.

`np.roll` is the vectorised `(n−k) % N`. The FFT version replaces `O(N²)` operations with `O(N log N)` — for `N = 4` that's irrelevant, but for filtering a 10-minute ECG at 500 Hz it's the difference between milliseconds and minutes.

`np.real(...)` is needed because floating-point round-off leaves tiny imaginary parts (like `1e-17j`) even though the true answer is real.

## 5.6 The key insight: circular = linear, wrapped

The linear convolution of the same signals is length 7:

```python
np.convolve([1,2,3,4], [1,0,2,1])    # [1, 2, 5, 9, 8, 11, 4]
```

Now fold that back modulo 4 — add index 4 onto index 0, index 5 onto index 1, index 6 onto index 2:

```
linear  :  1   2   5   9  | 8  11   4
             +8  +11  +4  |
─────────────────────────────
circular:  9  13   9   9
```

✓ Exactly the circular result. **This is time-domain aliasing** — the tail of the linear convolution wraps around and corrupts the start. It's the same phenomenon as frequency aliasing, one domain over.

```python
lin = np.convolve(x, h)
aliased = np.zeros(4)
for i, v in enumerate(lin):
    aliased[i % 4] += v
print(aliased)          # [ 9. 13.  9.  9.]
```

---

# Topic 6 — Linear Convolution via Circular Convolution

## 6.1 The idea

Since aliasing is caused by the output being longer than `N`, the fix is obvious: **make `N` big enough that there's no tail to wrap.**

> Zero-pad both signals to length `N ≥ len(x) + len(h) − 1`, then circular convolution equals linear convolution.

This is the basis of every fast convolution routine in existence, including `scipy.signal.fftconvolve`.

## 6.2 Solved example

Same `x = {1,2,3,4}`, `h = {1,0,2,1}`. Required `N ≥ 4 + 4 − 1 = 7`.

```python
def circconv_numpy_fft(x, h, N):
    return np.real(np.fft.ifft(np.fft.fft(x, N) * np.fft.fft(h, N)))

circconv_numpy_fft(x, h, 4)   # [ 9. 13.  9.  9.]           WRONG - aliased
circconv_numpy_fft(x, h, 7)   # [ 1.  2.  5.  9.  8. 11.  4.]  correct
np.convolve(x, h)             # [ 1.  2.  5.  9.  8. 11.  4.]  matches
```

Note `np.fft.fft(x, N)` — passing `N` zero-pads automatically. You don't need `np.pad`.

In practice you'd round `N` up to a power of two (8 here) for FFT speed; the extra zeros are harmless.

## 6.3 When to use which

| Situation | Use |
|---|---|
| Short kernel (`M < ~50`), any signal length | `np.convolve` — direct is faster |
| Long kernel, long signal | FFT convolution — `scipy.signal.fftconvolve` |
| Streaming / real-time, signal arrives in blocks | overlap-add or overlap-save |
| You genuinely want periodic behaviour | circular convolution, no padding |

---

# Topic 7 — Correlation

## 7.1 The idea

Convolution asks *"what does this system do to this signal?"* Correlation asks *"how similar are these two signals, and at what time offset?"*

$$r_{xy}[l] = \sum_{n} x[n]\,y[n-l]$$

`l` is the **lag** — how far you slide `y` before comparing. The formula is identical to convolution except **there is no flip**. That's the whole difference.

- **Cross-correlation** `r_xy` — two different signals. Peak location tells you the delay between them.
- **Autocorrelation** `r_xx` — a signal with itself. Reveals periodicity, and `r_xx[0]` equals the signal's energy.

## 7.2 Lag range and output length

`len(r) = len(x) + len(y) − 1`, and the lags run from `−(len(y)−1)` to `+(len(x)−1)`.

**NumPy does not give you the lag axis.** You must construct it:

```python
lags = np.arange(-(len(y) - 1), len(x))
```

Getting this wrong means your delay estimate is off by a constant — a bug that survives testing because the shape of the correlation looks perfectly fine.

## 7.3 Solved example — cross-correlation

`x = {1, 2, 3, 4}` (n=0…3), `y = {2, 1, 0, 1}` (n=0…3).

Length = 4+4−1 = 7, lags = −3 … +3.

Compute `r[l] = Σₙ x[n]·y[n−l]`, keeping only terms where both indices are in range:

```
l = -3:  x[0]y[3]                                   = 1·1                 = 1
l = -2:  x[0]y[2] + x[1]y[3]                        = 1·0 + 2·1           = 2
l = -1:  x[0]y[1] + x[1]y[2] + x[2]y[3]             = 1·1 + 2·0 + 3·1     = 4
l =  0:  x[0]y[0] + x[1]y[1] + x[2]y[2] + x[3]y[3]  = 1·2+2·1+3·0+4·1     = 8
l = +1:  x[1]y[0] + x[2]y[1] + x[3]y[2]             = 2·2 + 3·1 + 4·0     = 7
l = +2:  x[2]y[0] + x[3]y[1]                        = 3·2 + 4·1           = 10
l = +3:  x[3]y[0]                                   = 4·2                 = 8
```

**Answer:** `r_xy = {1, 2, 4, 8, 7, 10, 8}` for lags `−3…+3`. Peak at `l = +2`.

## 7.4 Solved example — autocorrelation

`x = {1, 2, 3, 4}` with itself:

```
l = -3:  1·4                        = 4
l = -2:  1·3 + 2·4                  = 11
l = -1:  1·2 + 2·3 + 3·4            = 20
l =  0:  1·1 + 2·2 + 3·3 + 4·4      = 30    <- maximum, = energy
l = +1:  2·1 + 3·2 + 4·3            = 20
l = +2:  3·1 + 4·2                  = 11
l = +3:  4·1                        = 4
```

**Answer:** `{4, 11, 20, 30, 20, 11, 4}` — symmetric about lag 0, maximum at lag 0. Both of those are always true for autocorrelation, so they're your free correctness check.

## 7.5 Loop program

```python
def xcorr_loop(x, y):
    """r_xy[l] = sum_n x[n]*y[n-l],  l = -(M-1) ... (N-1)"""
    N, M = len(x), len(y)
    lags = list(range(-(M - 1), N))
    r = []
    for l in lags:
        s = 0.0
        for n in range(N):
            if 0 <= n - l < M:            # both indices must be valid
                s += x[n] * y[n - l]
        r.append(s)
    return r, lags


def autocorr_loop(x):
    return xcorr_loop(x, x)
```

Compare this with `conv_loop` from Topic 3. The only structural difference is `y[n - l]` here versus `h[n - k]` there, with the loop variable playing a different role — no flip.

## 7.6 NumPy program

```python
x = np.array([1., 2., 3., 4.])
y = np.array([2., 1., 0., 1.])

r    = np.correlate(x, y, mode='full')          # [ 1.  2.  4.  8.  7. 10.  8.]
lags = np.arange(-(len(y) - 1), len(x))         # [-3 -2 -1  0  1  2  3]

# Equivalent, via convolution with the folded signal:
r2 = np.convolve(x, y[::-1])                    # identical

rxx = np.correlate(x, x, mode='full')           # [ 4. 11. 20. 30. 20. 11.  4.]
```

## 7.7 Correlation vs convolution — the exact relationship

$$r_{xy}[l] = x[l] * y[-l]$$

Correlation is convolution with one signal folded. Which means:

- If `y` is **symmetric** (`y == y[::-1]`), correlation and convolution give *identical* results
- Otherwise they differ, and the difference is a reversal

```python
x = np.array([1., 2., 3.]); h = np.array([1., 2.])
np.convolve(x, h)                  # [1. 4. 7. 6.]
np.correlate(x, h, 'full')         # [2. 5. 8. 3.]
np.convolve(x, h[::-1])            # [2. 5. 8. 3.]   <- matches correlation
```

## 7.8 Properties of correlation

| Property | Statement |
|---|---|
| Not commutative | `r_xy[l] = r_yx[−l]` — order matters, unlike convolution |
| Autocorr is even | `r_xx[l] = r_xx[−l]` |
| Autocorr peaks at 0 | `r_xx[0] ≥ \|r_xx[l]\|` for all `l` |
| Autocorr at 0 = energy | `r_xx[0] = Σ x[n]²` |
| Bound | `\|r_xy[l]\| ≤ √(r_xx[0]·r_yy[0])` (Cauchy–Schwarz) |

## 7.9 Normalised cross-correlation

Raw correlation values depend on signal amplitude, so you can't compare them across recordings. Normalising by mean and energy gives a value in `[−1, +1]`, where `+1` means "identical shape":

```python
def norm_xcorr(x, y):
    xm, ym = x - x.mean(), y - y.mean()          # remove DC offset
    r = np.correlate(xm, ym, 'full') / np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    return r, np.arange(-(len(y) - 1), len(x))

a = np.array([1., 2., 3., 4., 5.])
b = 10*a + 3                                     # different scale AND offset
r, lags = norm_xcorr(a, b)
print(r.max(), lags[np.argmax(r)])               # 1.0 at lag 0
```

Scale and offset are removed, so the peak is exactly 1.0 — the shapes match perfectly.

## 7.10 The main application: delay estimation

This is why correlation earns its place. Given a known reference and a noisy, delayed copy, the correlation peak recovers the delay even at poor SNR:

```python
rng = np.random.default_rng(0)
fs  = 500.0
t   = np.arange(0, 1, 1/fs)
sig = np.sin(2*np.pi*7*t) * np.exp(-3*t)         # a transient pulse

true_delay = 60                                   # samples
noisy = np.roll(sig, true_delay) + 0.5*rng.standard_normal(len(sig))

c    = np.correlate(noisy, sig, 'full')
lags = np.arange(-(len(sig) - 1), len(noisy))
print(lags[np.argmax(c)])                         # 59  (true 60)
```

One sample off at 500 Hz is a 2 ms error, with noise at half the signal amplitude. That robustness is why correlation underpins radar, sonar, GPS, and — in your case — template matching for QRS detection and template subtraction for maternal ECG cancellation.

---

# Topic 8 — The DFT

## 8.1 The idea

The DFT expresses an `N`-sample signal as a sum of `N` complex exponentials. It answers: *which frequencies are present, and with what amplitude and phase?*

$$X[k] = \sum_{n=0}^{N-1} x[n]\,e^{-j2\pi kn/N}, \qquad k = 0,1,\dots,N-1$$

$$x[n] = \frac{1}{N}\sum_{k=0}^{N-1} X[k]\,e^{+j2\pi kn/N}$$

The inverse differs by the sign in the exponent and the `1/N`. That's all.

Bin `k` corresponds to frequency `k·fs/N` Hz. Bin 0 is DC (the mean).

Writing `W_N = e^{-j2π/N}` (the "twiddle factor"), the DFT is `X[k] = Σ x[n]·W_N^{kn}`. For `N=4`: `W₄ = e^{-jπ/2} = −j`, so the powers cycle `1, −j, −1, +j`.

## 8.2 Solved example

`x = {1, 2, 3, 4}`, `N = 4`. Powers of `W₄ = −j` cycle with period 4: `W⁰=1, W¹=−j, W²=−1, W³=+j`.

```
X[0] = 1·1 + 2·1 + 3·1 + 4·1                = 10
X[1] = 1·1 + 2·(−j) + 3·(−1) + 4·(+j)
     = 1 − 2j − 3 + 4j                      = −2 + 2j
X[2] = 1·1 + 2·(−1) + 3·1 + 4·(−1)
     = 1 − 2 + 3 − 4                        = −2
X[3] = 1·1 + 2·(+j) + 3·(−1) + 4·(−j)
     = 1 + 2j − 3 − 4j                      = −2 − 2j
```

**Answer:** `X = {10, −2+2j, −2, −2−2j}`

Magnitudes: `|X| = {10, 2.828, 2, 2.828}` where `2.828 = 2√2`.

**Two free checks:**
- `X[0] = Σx[n] = 1+2+3+4 = 10` ✓ (DC is always the sum)
- For real input, `X[N−k] = conj(X[k])`. Here `X[3] = −2−2j = conj(−2+2j) = conj(X[1])` ✓

That conjugate symmetry is why `np.fft.rfft` exists — for real signals the second half is redundant.

## 8.3 Loop program

```python
import cmath

def dft_loop(x):
    N = len(x)
    X = []
    for k in range(N):
        real, imag = 0.0, 0.0
        for n in range(N):
            angle = -2 * cmath.pi * k * n / N
            real += x[n] * cmath.cos(angle).real
            imag += x[n] * cmath.sin(angle).real
        X.append(complex(round(real, 10), round(imag, 10)))
    return X


def idft_loop(X):
    N = len(X)
    xr = []
    for n in range(N):
        s = 0j
        for k in range(N):
            s += X[k] * cmath.exp(2j * cmath.pi * k * n / N)   # note: +2j
        xr.append((s / N).real)                                 # note: /N
    return xr
```

The `round(..., 10)` is cosmetic — it turns `-2.0000000000000004` into `-2.0` for readable output. The `.real` at the end of the IDFT is safe because the input was real; keeping tiny imaginary residue would just be noise.

## 8.4 NumPy program

```python
def dft_matrix(x):
    """Build the N x N DFT matrix and multiply. O(N^2) but fully vectorised."""
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)                     # column vector -> broadcasts to N x N
    W = np.exp(-2j * np.pi * k * n / N)      # W[k,n] = e^{-j2*pi*k*n/N}
    return W @ x

x = np.array([1., 2., 3., 4.])
print(dft_matrix(x))              # [10.+0.j -2.+2.j -2.-0.j -2.-2.j]
print(np.fft.fft(x))              # [10.+0.j -2.+2.j -2.+0.j -2.-2.j]
print(np.abs(np.fft.fft(x)))      # [10.  2.8284  2.  2.8284]
print(np.angle(np.fft.fft(x)))    # [0.  2.3562  3.1416  -2.3562]  radians
print(np.real(np.fft.ifft(np.fft.fft(x))))   # [1. 2. 3. 4.]
```

**What changed:** the double loop over `k` and `n` becomes an outer product. `k = n.reshape(-1,1)` makes `k` a column and `n` a row, so `k * n` broadcasts into the full `N×N` grid of products — one line replacing two nested loops. Then `W @ x` does all the sums at once.

`np.fft.fft` uses the FFT algorithm: `O(N log N)` instead of `O(N²)`. For `N = 4096` that's roughly a 340× speedup, which is why nobody uses the matrix version in production.

## 8.5 Reading a real spectrum

For an actual signal you need the frequency axis and correct amplitude scaling:

```python
fs = 1000.0
t  = np.arange(0, 1, 1/fs)
x  = 1.0*np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t)

N     = len(x)
X     = np.fft.rfft(x)                  # real FFT: only the useful half
freqs = np.fft.rfftfreq(N, d=1/fs)      # matching axis in Hz
mag   = 2 * np.abs(X) / N               # scale to true amplitude
```

The `2/N` scaling: `N` undoes the DFT's summation, and the `2` accounts for the energy that sat in the discarded negative-frequency half. With this, the peaks read exactly 1.000 at 50 Hz and 0.500 at 120 Hz — the amplitudes you put in.

Resolution is `fs/N` Hz. To resolve two close frequencies you need a **longer recording**, not a higher sampling rate.

## 8.6 Two theorems worth knowing

**Parseval** — energy is conserved across domains:

$$\sum_{n=0}^{N-1}|x[n]|^2 = \frac{1}{N}\sum_{k=0}^{N-1}|X[k]|^2$$

```python
X = np.fft.fft(x)
np.sum(x**2)                      # 30.0
np.sum(np.abs(X)**2) / len(x)     # 30.0
```

**Convolution theorem** — convolution in time is multiplication in frequency:

```python
a = np.array([1., 2., 3., 4.]); b = np.array([1., 2., 3.])
L = len(a) + len(b) - 1                      # zero-pad to avoid aliasing!
np.real(np.fft.ifft(np.fft.fft(a, L) * np.fft.fft(b, L)))   # [1. 4. 10. 16. 17. 12.]
np.convolve(a, b)                                            # [1. 4. 10. 16. 17. 12.]
```

Everything in Topics 3, 5, and 6 collapses into this one statement.

---

# Cheat Sheet

## Formulas

| Operation | Formula | Output length |
|---|---|---|
| Linear convolution | `y[n] = Σₖ x[k]h[n−k]` | `N + M − 1` |
| Circular convolution | `y[n] = Σₖ x[k]h[(n−k) mod N]` | `N` |
| Cross-correlation | `r_xy[l] = Σₙ x[n]y[n−l]` | `N + M − 1` |
| DFT | `X[k] = Σₙ x[n]e^{−j2πkn/N}` | `N` |
| IDFT | `x[n] = (1/N)Σₖ X[k]e^{+j2πkn/N}` | `N` |

## Loop → NumPy translation

| Loop pattern | NumPy equivalent |
|---|---|
| `for v in x: out.append(a*v)` | `a * x` |
| `if cond: 1 else: 0` per element | `(cond).astype(float)` |
| `if cond: a else: b` per element | `np.where(cond, a, b)` |
| `y[k:k+M] += x[k]*h` inner loop | slice assignment (one loop saved) |
| `(n-k) % N` wraparound | `np.roll(h, k)` |
| double loop over `k, n` | `n.reshape(-1,1) * n` broadcast |
| `sum(a[i]*b[i] for i in ...)` | `a @ b` or `np.dot(a, b)` |
| reverse a list | `x[::-1]` |
| nested loop matrix multiply | `H @ x` |

## Traps

1. **`x[n−2]` is a delay** — the signal moves *right*.
2. **NumPy returns values, not index axes.** Convolution output starts at `n₁+n₂`; correlation lags start at `−(len(y)−1)`. You must build these yourself.
3. **Circular ≠ linear** unless you zero-pad to `N ≥ len(x)+len(h)−1`.
4. **`mode='same'` has contaminated edges** — the first and last `M//2` samples are affected by zero-padding.
5. **Correlation is not commutative.** `r_xy[l] = r_yx[−l]`.
6. **`np.real()` after `ifft`** — round-off leaves imaginary crumbs.
7. **Spectrum scaling is `2/N`** for `rfft`, and frequency resolution `fs/N` improves with duration, not sample rate.

## Running checks

- Convolution: `Σy = Σx · Σh`
- Autocorrelation: symmetric, maximum at lag 0, `r_xx[0] = Σx²`
- DFT: `X[0] = Σx`, and for real input `X[N−k] = conj(X[k])`
- Any operation: loop version and NumPy version must agree to `np.allclose`

---

# Where This Leads

Every one of these operations shows up directly in fetal ECG extraction:

- **Template subtraction** — build an average maternal beat by aligning detected R-peaks (correlation), then subtract it at each occurrence (shifted impulse convolution)
- **QRS detection** — matched filtering is correlation with a beat template
- **Adaptive filters (LMS, RLS)** — the filter output is convolution; the weight update is driven by correlation between the error and the input
- **FastICA** — whitening is a covariance operation, which is autocorrelation at lag 0 generalised to multiple channels

The next natural topics are z-transform and pole-zero analysis, FIR design by windowing, IIR design via bilinear transform, and the overlap-add/overlap-save methods for filtering signals too long to hold in memory.
