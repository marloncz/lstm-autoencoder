import numpy as np
import pandas as pd


def simulate_ecg_data(n_beats: int, fs: int = 500) -> pd.DataFrame:
    """Simulating ECG signal using a sum of Gaussian functions for P, QRS, and T waves.

    Args:
        n_beats: Number of heart beats to simulate.
        fs: Sampling frequency of the ECG signal. Default is 500 Hz.

    Returns:
        DataFrame containing the ECG signal with columns "time" and "ecg_amplitude".
    """
    # time series for one heart beat
    t_beat = np.linspace(0, 1, fs, endpoint=False)
    ecg_signal = np.zeros(n_beats * fs)

    # Defining ECG components
    # P Wave
    def p_wave(t):
        a_p = 0.25
        t_p = 0.2  # P wave peak occurs at 0.2s
        sigma_p = 0.05
        return a_p * np.exp(-((t - t_p) ** 2) / (2 * sigma_p**2))

    # QRS Wave
    def qrs_complex(t):
        a_q = -0.15
        a_r = 1.0
        a_s = -0.15
        t_q = 0.1
        t_r = 0.2
        t_s = 0.3
        sigma_q = 0.02
        sigma_r = 0.01
        sigma_s = 0.02
        return (
            a_q * np.exp(-((t - t_q) ** 2) / (2 * sigma_q**2))
            + a_r * np.exp(-((t - t_r) ** 2) / (2 * sigma_r**2))
            + a_s * np.exp(-((t - t_s) ** 2) / (2 * sigma_s**2))
        )

    # T Wave
    def t_wave(t):
        a_t = 0.2
        t_t = 0.4
        sigma_t = 0.05
        return a_t * np.exp(-((t - t_t) ** 2) / (2 * sigma_t**2))

    # generating the ECG signal for n_beats
    for i in range(n_beats):
        ecg_signal[i * fs : (i + 1) * fs] = p_wave(t_beat) + qrs_complex(t_beat) + t_wave(t_beat)

    # time series for the entire ECG signal
    total_time = np.linspace(0, n_beats, n_beats * fs, endpoint=False)

    # defining DataFrame
    ecg_df = pd.DataFrame({"time": total_time, "ecg_amplitude": ecg_signal})

    return ecg_df
