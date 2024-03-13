import torch
import os


def save_checkpoint(state, full_file_path):
    # Save the check point if the model it best.
    create_file_if_does_not_exist(full_file_path)

    print("Saving a new best at file path: {}".format(full_file_path))
    torch.save(state, full_file_path)


def create_file_if_does_not_exist(full_file_path):
    # Create
    if not os.path.exists(full_file_path):
        model_tar = open(full_file_path, mode="w+")
        model_tar.close()
