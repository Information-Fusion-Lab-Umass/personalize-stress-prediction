import torch

from src import definitions
from src.bin import trainer
from src.models.multitask_learning import multitask_autoencoder
from src.utils import data_conversion_utils as conversions


def init_multitask_autoencoder_learner(num_features,
                                       autoencoder_bottle_neck_feature_size,
                                       autoencoder_num_layers,
                                       shared_hidden_layer_size,
                                       user_dense_layer_hidden_size,
                                       num_classes,
                                       num_covariates,
                                       shared_layer_dropout_prob,
                                       user_head_dropout_prob,
                                       learning_rate,
                                       decay,
                                       class_weights,
                                       student_list):
    class_weights = torch.tensor(class_weights)

    model = multitask_autoencoder.MultiTaskAutoEncoderLearner(
        conversions.prepend_ids_with_string(student_list, "student_"),
        num_features,
        autoencoder_bottle_neck_feature_size,
        autoencoder_num_layers,
        shared_hidden_layer_size,
        user_dense_layer_hidden_size,
        num_classes,
        num_covariates,
        shared_layer_dropout_prob,
        user_head_dropout_prob)

    if torch.cuda.is_available():
        model.cuda()
        class_weights = class_weights.cuda()

    reconstruction_criterion = torch.nn.L1Loss(reduction="sum")
    classification_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    return model, reconstruction_criterion, classification_criterion, optimizer


def get_params_from_model(params, data):
    use_histogram = params['use_histogram']
    autoencoder_bottle_neck_feature_size = params['autoencoder_bottle_neck_feature_size']
    autoencoder_num_layers = params['autoencoder_num_layers']
    alpha, beta = params['alpha'], params['beta']
    decay = params['decay']

    first_key = next(iter(data['data'].keys()))
    if use_histogram:
        num_features = len(data['data'][first_key][4][0])
    else:
        num_features = len(data['data'][first_key][0][0])

    num_covariates = len(data['data'][first_key][definitions.COVARIATE_DATA_IDX])
    shared_hidden_layer_size = params['shared_hidden_layer_size']
    user_dense_layer_hidden_size = params['user_dense_layer_hidden_size']
    num_classes = params['num_classes']
    learning_rate = params['learning_rate']
    n_epochs = params['n_epochs']
    shared_layer_dropout_prob = params["shared_layer_dropout_prob"]
    user_head_dropout_prob = params["user_head_dropout_prob"]
    class_weights = params['class_weights']
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

    return (use_histogram,
            autoencoder_bottle_neck_feature_size,
            autoencoder_num_layers,
            alpha, beta,
            decay,
            num_features,
            num_covariates,
            shared_hidden_layer_size,
            user_dense_layer_hidden_size,
            num_classes,
            learning_rate,
            n_epochs,
            shared_layer_dropout_prob,
            user_head_dropout_prob,
            class_weights,
            device)


def train_for_one_epoch(data,
                        num_classes,
                        model,
                        reconstruction_criterion,
                        classification_criterion,
                        device,
                        optimizer,
                        alpha,
                        beta,
                        use_histogram):
    (train_total_loss,
     train_total_reconstruction_loss,
     train_total_classification_loss,
     train_labels,
     train_preds,
     train_users) = trainer.evaluate_multitask_learner(data,
                                                       'train_ids',
                                                       num_classes,
                                                       model,
                                                       reconstruction_criterion,
                                                       classification_criterion,
                                                       device,
                                                       optimizer=optimizer,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       use_histogram=use_histogram)

    (val_total_loss,
     val_total_reconstruction_loss,
     val_total_classification_loss,
     val_labels,
     val_preds,
     val_users) = trainer.evaluate_multitask_learner(data,
                                                     'val_ids',
                                                     num_classes,
                                                     model,
                                                     reconstruction_criterion,
                                                     classification_criterion,
                                                     device,
                                                     alpha=alpha,
                                                     beta=beta,
                                                     use_histogram=use_histogram)

    return (train_total_loss,
            train_total_reconstruction_loss,
            train_total_classification_loss,
            train_labels,
            train_preds,
            train_users), (val_total_loss,
                           val_total_reconstruction_loss,
                           val_total_classification_loss,
                           val_labels,
                           val_preds,
                           val_users)
