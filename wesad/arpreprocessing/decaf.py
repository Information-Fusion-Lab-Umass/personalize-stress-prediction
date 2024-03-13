from arpreprocessing.helpers import add_ecg_to_subject, add_valence_arousal_to_subject
from arpreprocessing.preprocessor import Preprocessor, load_gzipped_pkl
from arpreprocessing.signal import NoSuchSignal


class Decaf(Preprocessor):
    SUBJECTS_IDS = tuple(range(1, 31))
    CHANNELS_NAMES = ["ECG", "EMG", "EOG"] + [f"EMO_{i}" for i in range(22)]
    WINDOWS_SIZE_SEC = 50

    def __init__(self, logger, path):
        Preprocessor.__init__(self, logger, path, "DECAF", self.CHANNELS_NAMES, get_sampling)
        self.load_data()

    def get_subjects_ids(self):
        return self.SUBJECTS_IDS

    def load_data(self):
        for format_ in ["movies", "music"]:
            info = load_gzipped_pkl(f"{self._path}/{format_}_info_arr")
            ecg = load_gzipped_pkl(f"{self._path}/{format_}_ecg_arr")
            emg = load_gzipped_pkl(f"{self._path}/{format_}_EMG_arr")
            eog = load_gzipped_pkl(f"{self._path}/{format_}_EOG_arr")
            emo = load_gzipped_pkl(f"{self._path}/{format_}_EMO_arr")

            for i, sample_info in enumerate(info):
                subject_id = int(sample_info[0][1:])
                subject = self.subjects[subject_id - 1]

                add_ecg_to_subject(ecg[i], subject, self.WINDOWS_SIZE_SEC, get_sampling)

                subject.x[1].data.append(emg[i][-self.WINDOWS_SIZE_SEC * get_sampling("EMG"):])
                subject.x[2].data.append(eog[i][-self.WINDOWS_SIZE_SEC * get_sampling("EOG"):])

                for j in range(22):
                    subject.x[j + 3].data.append(emo[i][j, -self.WINDOWS_SIZE_SEC * get_sampling("EMO"):])

                add_valence_arousal_to_subject(sample_info, subject, threshold_for_arousal=2)


def get_sampling(channel_name: str):
    if channel_name in ["ECG", "EMG", "EOG"]:
        return 64
    if channel_name.startswith("EMO"):
        return 5
    raise NoSuchSignal(channel_name)
