from arpreprocessing.helpers import add_ecg_to_subject, add_valence_arousal_to_subject
from arpreprocessing.preprocessor import Preprocessor, load_gzipped_pkl
from arpreprocessing.signal import NoSuchSignal
from utils.utils import interpolate_for_length


class Ascertain(Preprocessor):
    SUBJECTS_IDS = tuple(range(1, 59))
    CHANNELS_NAMES = ["ECG_0", "ECG_1", "GSR"] + [f"EEG_{i}" for i in range(8)] + [f"EMO_{i}" for i in range(22)]
    WINDOWS_SIZE_SEC = 50

    def __init__(self, logger, path):
        Preprocessor.__init__(self, logger, path, "ASCERTAIN", self.CHANNELS_NAMES, get_sampling)
        self.load_data()

    def get_subjects_ids(self):
        return self.SUBJECTS_IDS

    def load_data(self):
        info = load_gzipped_pkl(f"{self._path}/info_arr")
        eeg = load_gzipped_pkl(f"{self._path}/EEG_arr")
        ecg = load_gzipped_pkl(f"{self._path}/ECG_arr")
        gsr = load_gzipped_pkl(f"{self._path}/GSR_arr")
        emo = load_gzipped_pkl(f"{self._path}/EMO_arr")

        for i, sample_info in enumerate(info):
            subject_id = int(sample_info[0][-2:])
            subject = self.subjects[subject_id - 1]

            if len(ecg[i][0]) / get_sampling("ECG") < self.WINDOWS_SIZE_SEC:
                continue

            add_ecg_to_subject(ecg[i][0], subject, self.WINDOWS_SIZE_SEC, get_sampling, signals_poz=0)
            add_ecg_to_subject(ecg[i][1], subject, self.WINDOWS_SIZE_SEC, get_sampling, signals_poz=1)
            gsr_len = self.WINDOWS_SIZE_SEC * get_sampling("GSR")
            subject.x[2].data.append(interpolate_for_length(gsr[i][-gsr_len:], gsr_len))

            for j in range(8):
                subject.x[j + 3].data.append(eeg[i][j, -self.WINDOWS_SIZE_SEC * get_sampling("EEG"):])

            for j in range(22):
                subject.x[j + 11].data.append(emo[i][j, -self.WINDOWS_SIZE_SEC * get_sampling("EMO"):])

            add_valence_arousal_to_subject(sample_info, subject, threshold_for_arousal=3)


def get_sampling(channel_name: str):
    if channel_name.startswith("ECG"):
        return 64
    if channel_name == "GSR":
        return 8
    if channel_name.startswith("EEG"):
        return 32
    if channel_name.startswith("EMO"):
        return 5
    raise NoSuchSignal(channel_name)
