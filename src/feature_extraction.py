"""
MEA Feature Extraction — Firing Rate Computation from Raw Electrophysiology
Author: Adham Aboulkheir | University of Essex | PhD Research
"""
import numpy as np
from scipy import signal as sp_signal
from typing import List, Tuple


def bandpass_filter(data: np.ndarray, fs: float = 20000.0,
                    low: float = 300.0, high: float = 3000.0) -> np.ndarray:
    """
    Apply bandpass filter for spike detection (300-3000 Hz).
    
    Parameters
    ----------
    data : np.ndarray, shape (n_electrodes, n_samples)
    fs   : sampling frequency in Hz
    low  : lower cutoff frequency
    high : upper cutoff frequency
    
    Returns
    -------
    filtered : np.ndarray, same shape as input
    """
    nyq = fs / 2.0
    b, a = sp_signal.butter(4, [low / nyq, high / nyq], btype="band")
    return sp_signal.filtfilt(b, a, data, axis=-1)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR) to reduce common-mode noise.
    Subtracts the mean across all electrodes at each time point.
    """
    return data - data.mean(axis=0, keepdims=True)


def detect_spikes(data: np.ndarray, fs: float = 20000.0,
                  threshold_factor: float = 5.0,
                  refractory_period_ms: float = 1.0) -> List[np.ndarray]:
    """
    Detect spikes using adaptive threshold (threshold_factor × RMS noise).
    
    Parameters
    ----------
    data               : filtered MEA recording (n_electrodes, n_samples)
    fs                 : sampling frequency
    threshold_factor   : threshold = -factor × RMS
    refractory_period_ms : minimum time between spikes (ms)
    
    Returns
    -------
    spike_times : list of arrays, one per electrode (spike times in seconds)
    """
    refractory_samples = int(refractory_period_ms * fs / 1000)
    spike_times = []
    
    for electrode in range(data.shape[0]):
        trace = data[electrode]
        noise_rms = np.sqrt(np.mean(trace ** 2))
        threshold = -threshold_factor * noise_rms
        
        # Find threshold crossings (negative peaks)
        crossings = np.where((trace[:-1] > threshold) & (trace[1:] <= threshold))[0]
        
        # Apply refractory period
        if len(crossings) > 1:
            valid = [crossings[0]]
            for c in crossings[1:]:
                if c - valid[-1] >= refractory_samples:
                    valid.append(c)
            crossings = np.array(valid)
        
        spike_times.append(crossings / fs)
    
    return spike_times


def compute_firing_rates(spike_times: List[np.ndarray], duration: float,
                         window_size: float = 0.1) -> np.ndarray:
    """
    Compute firing rate matrix (electrodes × time windows).
    
    Parameters
    ----------
    spike_times : list of spike time arrays per electrode
    duration    : total recording duration (seconds)
    window_size : time window size (seconds)
    
    Returns
    -------
    firing_rates : (n_electrodes, n_windows) array in Hz
    """
    n_electrodes = len(spike_times)
    n_windows = int(duration / window_size)
    firing_rates = np.zeros((n_electrodes, n_windows))
    
    for e, times in enumerate(spike_times):
        for w in range(n_windows):
            t_start = w * window_size
            t_end = (w + 1) * window_size
            count = np.sum((times >= t_start) & (times < t_end))
            firing_rates[e, w] = count / window_size  # Hz
    
    return firing_rates


def extract_waveforms(data: np.ndarray, spike_times: List[np.ndarray],
                      fs: float = 20000.0, window_ms: float = 2.0) -> List[np.ndarray]:
    """
    Extract spike waveforms around detected spike times.
    
    Returns list of (n_spikes, n_samples) arrays per electrode.
    """
    window_samples = int(window_ms * fs / 1000)
    half = window_samples // 2
    waveforms = []
    
    for e, times in enumerate(spike_times):
        electrode_waveforms = []
        for t in times:
            idx = int(t * fs)
            if idx - half >= 0 and idx + half < data.shape[1]:
                electrode_waveforms.append(data[e, idx - half:idx + half])
        waveforms.append(np.array(electrode_waveforms) if electrode_waveforms else np.empty((0, window_samples)))
    
    return waveforms


if __name__ == "__main__":
    print("MEA Feature Extraction Demo")
    print("=" * 40)
    
    np.random.seed(42)
    fs = 20000
    duration = 10.0
    n_electrodes = 142
    n_samples = int(fs * duration)
    
    # Simulate raw MEA data
    raw_data = np.random.normal(0, 15, (n_electrodes, n_samples))
    
    # Add synthetic spikes to 20 electrodes
    for e in range(20):
        n_spikes = np.random.randint(50, 200)
        positions = np.random.choice(n_samples - 100, n_spikes, replace=False)
        for pos in positions:
            raw_data[e, pos:pos+10] += np.random.choice([-80, -60, -70]) * np.hanning(10)
    
    print(f"Raw data: {n_electrodes} electrodes × {n_samples} samples ({duration}s @ {fs}Hz)")
    
    # Process
    filtered = bandpass_filter(raw_data, fs=fs)
    car_data = common_average_reference(filtered)
    spike_times = detect_spikes(car_data, fs=fs)
    firing_rates = compute_firing_rates(spike_times, duration=duration)
    
    active = np.sum(firing_rates.mean(axis=1) > 1.0)
    total_spikes = sum(len(t) for t in spike_times)
    
    print(f"Total spikes detected: {total_spikes}")
    print(f"Active electrodes (>1 Hz): {active} / {n_electrodes}")
    print(f"Firing rates shape: {firing_rates.shape}")
    print(f"Mean firing rate: {firing_rates.mean():.2f} Hz")
    print(f"Max firing rate: {firing_rates.max():.2f} Hz")
