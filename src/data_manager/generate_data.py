from src.data_manager import student_life_var_binned_data_manager as data_manager
from src.bin import statistics
from src.utils import write_utils

student_list = [4, 7, 8, 10, 14, 16, 17, 19, 22, 23, 24, 32, 33, 35, 36, 43, 44, 49, 51, 52, 53, 57, 58]
data = data_manager.get_data_for_training_in_dict_format(*student_list, normalize=True, fill_na=True,
                                                         flatten_sequence=False, split_type='percentage')
print(statistics.get_train_test_val_label_counts_from_raw_data(data))

write_utils.data_structure_to_pickle(data,
                                     '../data/training_data/shuffled_splits/training_data_normalized_no_prev_stress_students_greater_than_40_labels.pkl')
