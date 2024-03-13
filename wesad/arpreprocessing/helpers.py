import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from sklearn.preprocessing import MinMaxScaler

from arpreprocessing.signal import NoSuchSignal


def add_ecg_to_subject(ecg, subject, window_size_sec, get_sampling_fn, signals_poz=0):
    ecg_ = ecg[-window_size_sec * get_sampling_fn("ECG"):]
    subject.x[signals_poz].data.append(ecg_)


def add_valence_arousal_to_subject(sample_info, subject, threshold_for_arousal=0, threshold_for_valence=0):
    arousal = 1 if sample_info[2] >= threshold_for_arousal else 0
    valence = 1 if sample_info[3] >= threshold_for_valence else 0
    subject.y.append(arousal * 2 + valence + 1)


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff=5, fs=64, order=3, start_from=1000):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return pd.Series(y[start_from:])


def filter_signal(signal_name, signal, original_sampling_fn, target_sampling_fn):
    result = scipy.stats.mstats.winsorize(signal, limits=[0.03, 0.03])
    if original_sampling_fn(signal_name) / 2 > 10:
        result = butter_lowpass_filter(result, cutoff=10, fs=original_sampling_fn(signal_name), start_from=0)

    result = pd.Series(result).iloc[::int(original_sampling_fn(signal_name) / target_sampling_fn(signal_name))]

    result = np.array(result).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(result)
    result = scaler.transform(result)

    return result.reshape(1, -1)[0]


def get_empatica_sampling(channel_name: str):
    if channel_name.startswith("BVP"):
        return 64
    if channel_name.startswith("ACC"):
        return 32
    if channel_name.startswith("EDA"):
        return 4
    if channel_name.startswith("TEMP"):
        return 4
    raise NoSuchSignal(channel_name)
