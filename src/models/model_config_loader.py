import torch
from src.utils import read_utils
from src import definitions
from src.bin import validations
from src.bin import statistics
from src.utils import data_conversion_utils as conversions


def load_static_configs_for_lstm_n_multitask_models(model):
    model_config = read_utils.read_yaml(definitions.MODEL_CONFIG_FILE_PATH)[
                                                        'lstm_n_multitask']
    validations.validate_config_key(model, config=model_config)

    # Global configs which are common to every model.
    use_histogram = model_config['use_histogram']
    autoencoder_bottle_neck_feature_size = model_config['autoencoder_bottle_neck_feature_size']
    autoencoder_num_layers = model_config['autoencoder_num_layers']
    shared_hidden_layer_size = model_config['shared_hidden_layer_size']
    user_dense_layer_hidden_size = model_config['user_dense_layer_hidden_size']
    num_classes = model_config['num_classes']
    decay = model_config['decay']
    shared_layer_dropout_prob = model_config['shared_layer_dropout_prob']
    user_head_dropout_prob = model_config['user_head_dropout_prob']

    # Specific configs that vary across models.
    alpha = model_config[model]['alpha']
    beta = model_config[model]['beta']
    learning_rate = model_config[model]['learning_rate']
    n_epochs = model_config[model]['n_epochs']
    bidirectional = model_config[model]['bidirectional']

    return (use_histogram,
            autoencoder_bottle_neck_feature_size,
            autoencoder_num_layers,
            shared_hidden_layer_size,
            user_dense_layer_hidden_size,
            num_classes,
            decay,
            shared_layer_dropout_prob,
            user_head_dropout_prob,
            alpha,
            beta,
            learning_rate,
            n_epochs,
            bidirectional)


def load_derived_configs_for_lstm_n_multitask_models(use_histogram, data):
    # Derived Configs
    first_key = next(iter(data['data'].keys()))
    if use_histogram:
        num_features = len(data['data'][first_key][4][0])
    else:
        num_features = len(data['data'][first_key][0][0])
    num_covariates = len(data['data'][first_key][definitions.COVARIATE_DATA_IDX])
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    class_weights = torch.tensor(statistics.get_class_weights_in_inverse_proportion(data))
    cuda_enabled = torch.cuda.is_available()
    student_list = conversions.extract_distinct_student_idsfrom_keys(data['data'].keys())

    return num_features, num_covariates, device, class_weights, cuda_enabled, student_list
