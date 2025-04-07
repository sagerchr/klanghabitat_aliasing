import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Streamlit setup
st.set_page_config(layout="centered", page_title="Aliasing Demo")

# Constants
Fs = 48000  # Sampling rate in Hz
nyquist = Fs / 2
duration = 0.01  # 10 ms total duration (for FFT accuracy)

st.title("🔁 Signal Aliasing Demo")
st.markdown(f"<h3>Sampling Rate: {Fs / 1000:.0f} kHz &nbsp;&nbsp;|&nbsp;&nbsp; Nyquist Frequency: {nyquist / 1000:.0f} kHz</h3>", unsafe_allow_html=True)

# UI controls
waveform = st.selectbox("Waveform Type", ["Sine", "Square", "Triangle"])
signal_khz = st.slider("Signal Frequency (kHz)", min_value=10.0, max_value=40.0, step=0.1, value=10.0)
signal_hz = signal_khz * 1000

# Time vectors
t_cont = np.linspace(0, duration, 100000)     # for smooth plot
t_samp = np.arange(0, duration, 1.0 / Fs)     # for sampling

# Generate original and sampled signals
if waveform == "Sine":
    signal_true = np.sin(2 * np.pi * signal_hz * t_cont)
    samples = np.sin(2 * np.pi * signal_hz * t_samp)
elif waveform == "Square":
    signal_true = np.sign(np.sin(2 * np.pi * signal_hz * t_cont))
    samples = np.sign(np.sin(2 * np.pi * signal_hz * t_samp))
elif waveform == "Triangle":
    signal_true = 2 * np.abs(2 * (t_cont * signal_hz % 1) - 1) - 1
    samples = 2 * np.abs(2 * (t_samp * signal_hz % 1) - 1) - 1

# Sinc reconstruction
def sinc_reconstruct(samples, t_samp, t_recon):
    T = t_samp[1] - t_samp[0]
    y = np.zeros_like(t_recon)
    for i in range(len(t_samp)):
        y += samples[i] * np.sinc((t_recon - t_samp[i]) / T)
    return y

recon = sinc_reconstruct(samples, t_samp, t_cont)

# Aliased frequency
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

# Time Domain - Original Signal
axs[0].plot(t_cont, signal_true, color='gray', alpha=0.5, label="Original Signal")
axs[0].plot(t_samp, samples, 'ko', markersize=3, label="Sampled Points")
axs[0].set_title(f"Original Signal | {signal_khz:.2f} kHz ({waveform})")
axs[0].set_ylim(-1.2, 1.2)
axs[0].set_xlim(0, 0.001)  # Show only 1 ms
axs[0].grid(True)

# Time Domain - Reconstructed
axs[1].plot(t_cont, recon, color='green', label="Reconstructed Signal")
axs[1].plot(t_samp, samples, 'ko', markersize=3)
axs[1].set_title(f"Reconstructed Signal | Appears as {alias_freq / 1000:.2f} kHz")
axs[1].set_ylim(-1.2, 1.2)
axs[1].set_xlim(0, 0.001)
axs[1].set_xlabel("Time (s)")
axs[1].grid(True)

# Frequency Domain - Spectrum
axs[2].semilogx(fft_freqs, 20 * np.log10(spectrum + 1e-12), color='purple')
axs[2].axvline(nyquist, linestyle='--', color='red', label="Nyquist Limit (24 kHz)")
axs[2].set_title("Spectrum of Sampled Signal (Log Frequency Scale)")
axs[2].set_xlabel("Frequency (Hz)")
axs[2].set_ylabel("Magnitude (dB)")
axs[2].set_xlim(10, Fs / 2 + 2000)
axs[2].set_ylim(-80, 0)
axs[2].legend()
axs[2].grid(True, which='both', linestyle='--', alpha=0.5)

# Show plot
plt.tight_layout()
plt.subplots_adjust(hspace=0.4, bottom=0.07)
st.pyplot(fig)
