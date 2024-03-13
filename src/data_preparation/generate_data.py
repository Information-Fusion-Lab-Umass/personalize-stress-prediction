from src.utils import student_life_var_binned_data_manager as data_manager
import pickle

# student_list = [4, 7, 8, 10, 14, 16, 17, 19, 22, 23, 24, 32, 33, 35, 36, 43, 44, 49, 51, 52, 53, 57, 58]
student_list = [4, 7, 8, 10, 14, 16, 17, 19, 22, 23, 24, 32, 33, 35, 36, 43, 44, 49, 51, 52, 53, 57, 58]
data = data_manager.get_data_for_training_in_dict_format(*student_list, normalize=True, fill_na=True,
                                                         flatten_sequence=False, split_type='percentage')

print(len(data))

# # save
# saved_filename = '../data/training_data/shuffled_splits/training_data_normalized_no_prev_stress_students_greater_than_40_labels.pkl'
# with open(saved_filename, 'wb') as f:
#     pickle.dump(data, f)