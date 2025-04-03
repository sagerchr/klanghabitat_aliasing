import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit page setup
st.set_page_config(layout="centered", page_title="Aliasing Demo")

# Sampling settings
Fs = 48000  # Hz
nyquist = Fs / 2
duration = 0.01  # seconds

st.title("🔁 Signal Aliasing Demo")
st.markdown(f"<h3>Sampling Rate: {Fs // 1000} kHz &nbsp;&nbsp;|&nbsp;&nbsp; Nyquist Frequency: {nyquist / 1000:.0f} kHz</h3>", unsafe_allow_html=True)

# Frequency selection
signal_khz = st.slider("Signal Frequency (kHz)", min_value=10.0, max_value=40.0, step=0.1, value=10.0)
signal_hz = signal_khz * 1000

# Reconstruction using sinc interpolation
def sinc_reconstruct(samples, t_samp, t_recon):
    T = t_samp[1] - t_samp[0]
    y = np.zeros_like(t_recon)
    for i in range(len(t_samp)):
        y += samples[i] * np.sinc((t_recon - t_samp[i]) / T)
    return y

# Time vectors
t_cont = np.linspace(0, duration, 100000)
t_samp = np.arange(0, duration, 1.0 / Fs)

# Generate signals
signal_true = np.sin(2 * np.pi * signal_hz * t_cont)
samples = np.sin(2 * np.pi * signal_hz * t_samp)
recon = sinc_reconstruct(samples, t_samp, t_cont)

# Compute alias frequency
if signal_hz <= nyquist:
    alias_freq = signal_hz
else:
    alias_freq = abs(signal_hz - Fs * round(signal_hz / Fs))

st.markdown(f"**Reconstructed / Aliased Frequency:** {alias_freq / 1000:.2f} kHz")

# Spectrum of sampled signal
N = len(samples)
fft_vals = np.fft.fft(samples)
fft_freqs = np.fft.fftfreq(N, 1.0 / Fs)
spectrum = np.abs(fft_vals) / N
spectrum = spectrum[:N // 2]
fft_freqs = fft_freqs[:N // 2]

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=False)

# Plot 1: Original Signal
axs[0].plot(t_cont, signal_true, color='gray', alpha=0.5, label="Original Signal")
axs[0].plot(t_samp, samples, 'ko', markersize=3, label="Sampled Points")
axs[0].set_title(f"Original Signal | {signal_khz:.2f} kHz Input")
axs[0].set_ylim(-1.2, 1.2)
axs[0].grid(True)

# Plot 2: Reconstructed Signal
axs[1].plot(t_cont, recon, color='green', label="Reconstructed Signal")
axs[1].plot(t_samp, samples, 'ko', markersize=3)
axs[1].set_title(f"Reconstructed Signal | Appears as {alias_freq / 1000:.2f} kHz")
axs[1].set_ylim(-1.2, 1.2)
axs[1].set_xlabel("Time (s)")
axs[1].grid(True)

# Plot 3: Spectrum
axs[2].semilogx(fft_freqs, 20 * np.log10(spectrum + 1e-12), color='purple')
axs[2].axvline(nyquist, linestyle='--', color='red', label="Nyquist Limit (24 kHz)")
axs[2].set_title("Spectrum of Sampled Signal (Log Frequency Scale)")
axs[2].set_xlabel("Frequency (Hz)")
axs[2].set_ylabel("Magnitude (dB)")
axs[2].set_xlim(10, Fs / 2 + 2000)
axs[2].set_ylim(-80, 0)
axs[2].legend()
axs[2].grid(True, which='both', linestyle='--', alpha=0.5)

axs[0].set_xlim(0, 0.0001)  # show only the first 1 ms
axs[1].set_xlim(0, 0.0001)  # same for reconstructed

plt.tight_layout()
plt.subplots_adjust(hspace=0.4, bottom=0.07)
st.pyplot(fig)
