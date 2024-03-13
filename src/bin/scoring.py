from sklearn import metrics


def get_precission_recall_f_scores(**kwargs):
    """

    @param kwargs:
    @return: Get scores
    """

    loss_over_epochs = kwargs.get('loss_over_epochs', None)
    scores_over_epochs = kwargs.get('scores_over_epochs', None)

    train_loss = kwargs.get('train_loss', None)
    train_labels = kwargs.get('train_labels', None)
    train_preds = kwargs.get('train_preds', None)

    val_loss = kwargs.get('val_loss', None)
    val_labels = kwargs.get('val_labels', None)
    val_preds = kwargs.get('val_preds', None)

    test_loss = kwargs.get('test_loss', None)
    test_labels = kwargs.get('test_labels', None)
    test_preds = kwargs.get('test_preds', None)

    average = kwargs.get('average', 'macro')

    train_scores = metrics.precision_recall_fscore_support(train_labels,
                                                           train_preds,
                                                           average=average)
    val_scores = metrics.precision_recall_fscore_support(val_labels,
                                                         val_preds,
                                                         average=average)
    test_scores = metrics.precision_recall_fscore_support(test_labels,
                                                          test_preds,
                                                          average=average)

    return train_scores, val_scores, test_scores
