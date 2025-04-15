import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Sample parameters
fs = 44100        # Sampling frequency
T = 1.0           # Duration in seconds
f0 = 1000.0       # Grundfrequenz 1 kHz
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Eingangssignal (Sinus)
x = np.sin(2 * np.pi * f0 * t)

# Saturation-Funktionen
def soft_clip_very_strong(signal):
    return signal - (1/3) * signal**3 + (1/5) * signal**5 - (1/7) * signal**7

def hard_clip(signal, threshold=0.2):
    return np.clip(signal, -threshold, threshold)

def tanh_clip(signal, gain=5):
    return np.tanh(gain * signal)

def diode_clip(signal):
    return np.maximum(0, signal) - 0.5 * np.maximum(0, signal - 0.4)

# Verarbeitete Signale
signals = {
    "Soft Clipping (sehr stark)": soft_clip_very_strong(x),
    "Hard Clipping": hard_clip(x),
    "Tanh Clipping": tanh_clip(x),
    "Diode Clipping": diode_clip(x)
}

# FFT-Funktion
def compute_fft(signal, fs):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    return xf[:N//2], 2.0/N * np.abs(yf[:N//2])

# FFT-Daten berechnen
fft_data = {name: compute_fft(sig, fs) for name, sig in signals.items()}

# Normieren auf 1 kHz-Peak
def normalize_fft(xf, yf, freq=1000.0, tolerance=50):
    mask = (xf >= freq - tolerance) & (xf <= freq + tolerance)
    peak = np.max(yf[mask])
    return yf / peak if peak != 0 else yf

fft_data_norm = {
    name: (xf, normalize_fft(xf, yf))
    for name, (xf, yf) in fft_data.items()
}

# Mathematische Formeln für Plot-Titel
formulas = {
    "Soft Clipping (sehr stark)": r"$x - \frac{1}{3}x^3 + \frac{1}{5}x^5 - \frac{1}{7}x^7$",
    "Hard Clipping": r"$\mathrm{clip}(x, -\theta, \theta)$",
    "Tanh Clipping": r"$\tanh(g \cdot x)$",
    "Diode Clipping": r"$\max(0, x) - 0.5 \cdot \max(0, x - 0.4)$"
}

# Plotten
plt.figure(figsize=(12, 14))
for i, (label, (xf, yf)) in enumerate(fft_data_norm.items(), 1):
    plt.subplot(5, 1, i)
    plt.semilogy(xf, yf)
    plt.title(f"{label} – {formulas.get(label, '')}")
    plt.xlim(0, 10000)
    plt.ylim(1e-3, 1.1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (log, normiert)")

plt.tight_layout()
plt.show()
