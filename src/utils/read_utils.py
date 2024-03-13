import yaml
import os
import pickle


def read_yaml(file_path):
    """Util to read Yaml File."""

    # Reading from YML file.
    with open(file_path, "r") as ymlfile:
        yaml_file = yaml.load(ymlfile)

    return yaml_file


def read_pickle(file_path):
    if not os.path.exists(file_path):
        raise "File as {} does not exist.".format(file_path)

    with (open(file_path, "rb")) as file:
        data = pickle.load(file)

    return data
