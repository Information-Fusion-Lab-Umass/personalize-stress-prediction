import math

from arpreprocessing.helpers import add_ecg_to_subject, add_valence_arousal_to_subject
from arpreprocessing.preprocessor import Preprocessor, load_gzipped_pkl
from arpreprocessing.signal import NoSuchSignal
from utils.utils import interpolate_for_length


class Amigos(Preprocessor):
    SUBJECTS_IDS = tuple(range(40))
    CHANNELS_NAMES = ["ECG_0", "ECG_1", "GSR"] + [f"EEG_{i}" for i in range(14)]
    THRESHOLD = 5
    WINDOWS_SIZE_SEC = 50

    def __init__(self, logger, path):
        Preprocessor.__init__(self, logger, path, "Amigos", self.CHANNELS_NAMES, get_sampling)
        self.load_data()

    def get_subjects_ids(self):
        return self.SUBJECTS_IDS

    def load_data(self):
        info = load_gzipped_pkl(f"{self._path}/info_arr")
        eeg = load_gzipped_pkl(f"{self._path}/EEG_arr")
        ecg = load_gzipped_pkl(f"{self._path}/ECG_arr")
        gsr = load_gzipped_pkl(f"{self._path}/GSR_arr")

        for i, sample_info in enumerate(info):
            if len(sample_info) < 14:
                continue

            subject_id = int(sample_info[0][1:3]) - 1
            subject = self.subjects[subject_id]

            if len(ecg[i][:, 0]) / get_sampling("ECG") < self.WINDOWS_SIZE_SEC:
                continue

            if len(ecg[i][:, 0]) / get_sampling("ECG") > (10 * 60):
                continue

            if math.isnan(ecg[i][:, 0][0]):
                continue

            add_ecg_to_subject(ecg[i][:, 0], subject, self.WINDOWS_SIZE_SEC, get_sampling, signals_poz=0)
            add_ecg_to_subject(ecg[i][:, 1], subject, self.WINDOWS_SIZE_SEC, get_sampling, signals_poz=1)
            gsr_len = self.WINDOWS_SIZE_SEC * get_sampling("GSR")
            subject.x[2].data.append(interpolate_for_length(gsr[i][-gsr_len:], gsr_len))

            for j in range(14):
                subject.x[j + 3].data.append(eeg[i][-self.WINDOWS_SIZE_SEC * get_sampling("EEG"):, 0])

            add_valence_arousal_to_subject(sample_info, subject, threshold_for_arousal=self.THRESHOLD,
                                           threshold_for_valence=self.THRESHOLD)


def get_sampling(channel_name: str):
    if channel_name.startswith("ECG"):
        return 64
    if channel_name.startswith("EEG"):
        return 64
    if channel_name == "GSR":
        return 8
    raise NoSuchSignal(channel_name)
