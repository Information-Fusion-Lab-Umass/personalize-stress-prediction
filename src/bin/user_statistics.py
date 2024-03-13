from sklearn import metrics
from src import definitions
from src.bin import validations


def user_confusion_matrix(user_data):
    """

    @param user_data: Data for single user
    @return: Confusion matrix as list of lists.
    """
    validations.validate_user_data(user_data)
    validations.validate_single_values_column_in_df(user_data, 'user')

    return metrics.confusion_matrix(user_data['label'], user_data['prediction'], labels=definitions.LABELS)


def user_f1_score(user_data):
    validations.validate_user_data(user_data)
    validations.validate_single_values_column_in_df(user_data, 'user')

    return metrics.f1_score(user_data['label'], user_data['prediction'], average='weighted')


def user_accuracy(user_data):
    validations.validate_user_data(user_data)
    validations.validate_single_values_column_in_df(user_data, 'user')

    return metrics.accuracy_score(user_data['label'], user_data['prediction'])


def label_count(user_data):
    validations.validate_user_data(user_data)
    validations.validate_single_values_column_in_df(user_data, 'user')

    return len(user_data['label'])
