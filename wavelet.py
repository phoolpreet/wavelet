# https://github.com/CSchoel/learn-wavelets/blob/main/wavelet-introduction.ipynb

import numpy as np
from matplotlib import pyplot as plt


def mexican_hat(x, mu, sigma):
    c1 = 2.0 / (np.sqrt(3 * sigma) * np.pi**0.25)
    c2 = 1.0 - ((x - mu) ** 2 / sigma**2)
    c3 = np.exp(-((x - mu) ** 2) / 2.0 * sigma**2)
    return c1 * c2 * c3


def gauss(x, mu, sigma):
    c1 = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    c2 = np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))
    return c1 * c2


def hamming(n):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def sound(freq, dur, res):
    ln = dur * res
    sound = np.sin(np.arange(ln) * 2 * np.pi * freq / res)
    sound = sound * hamming(ln)
    return sound


def add_sound(audio, loc, freq, dur, res):
    audio[int(loc) : int(loc + dur * res)] += sound(freq, dur, res)


res = 10000
audio = np.zeros(15000)
add_sound(audio, 1000, 100, 0.5, res)
add_sound(audio, 3000, 130, 0.5, res)
add_sound(audio, 2000, 50, 1, res)
add_sound(audio, 10000, 150, 0.5, res)

fourier = np.fft.fft(audio)
xvals = np.fft.fftfreq(len(audio)) * res
idx = np.where(np.abs(xvals) < 200)
plt.plot(xvals[idx], np.abs(fourier)[idx])
plt.show()
