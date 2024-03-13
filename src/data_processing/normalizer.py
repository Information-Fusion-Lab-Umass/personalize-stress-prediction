import pandas as pd


def normalize(data_frame: pd.DataFrame, norm_type="mean",
              df_mean: pd.Series = None, df_std: pd.Series = None,
              df_min: pd.Series = None, df_max: pd.Series = None) -> pd.DataFrame:
    if norm_type == "min_max":
        if df_min is None:
            df_min = data_frame.min()
        if df_max is None:
            df_max = data_frame.max()

        result = (data_frame - df_min) / (df_max - df_min)
    else:
        if df_mean is None:
            df_mean = data_frame.mean()
        if df_mean is None:
            df_std = data_frame.std()

        result = (data_frame - df_mean) / df_std

    return result.fillna(0)


def normalize_data_list(data_list: list, normalize_strat='mean'):
    """
    This function calculates the global mean, stdev, min, max and normalizes each data tuple based
    on the global statistic.

    @param data_list: Accepts a list of tuple with the first element being
                      month_day_hour_key and second being data tuple.
    @return: Normalized data list.
    """
    new_data_list = []
    global_train_values = pd.DataFrame()
    global_histogram = pd.DataFrame()
    global_covariates = pd.DataFrame()

    for month_day_hour_key, data_tuple in data_list:
        training_values, missing_values, time_delta, covariates, histogram, y_labels = data_tuple
        global_train_values = global_train_values.append(pd.DataFrame(training_values), ignore_index=True)
        global_histogram = global_histogram.append(pd.DataFrame(histogram), ignore_index=True)
        global_covariates = global_covariates.append(pd.DataFrame([covariates]), ignore_index=True)

    global_train_mean, global_train_std, global_train_min, global_train_max = evaluate_df_statistic(global_train_values)
    global_hist_mean, global_hist_std, global_hist_min, global_hist_max = evaluate_df_statistic(global_histogram)
    global_covariates_mean, global_covariates_std, global_covariates_min, global_covariates_max = evaluate_df_statistic(
        global_covariates)

    for month_day_hour_key, data_tuple in data_list:
        training_values, missing_values, time_delta, covariates, histogram, y_labels = data_tuple
        local_train_values = pd.DataFrame(training_values)
        local_histogram = pd.DataFrame(histogram)
        local_covariate = pd.DataFrame([covariates])

        local_train_values = normalize(local_train_values,
                                       norm_type=normalize_strat,
                                       df_mean=global_train_mean,
                                       df_std=global_train_std,
                                       df_min=global_train_min,
                                       df_max=global_train_max)
        local_histogram = normalize(local_histogram,
                                    norm_type=normalize_strat,
                                    df_mean=global_hist_mean,
                                    df_std=global_hist_std,
                                    df_min=global_hist_min,
                                    df_max=global_hist_max)

        local_covariate = normalize(local_covariate,
                                    norm_type=normalize_strat,
                                    df_mean=global_covariates_mean,
                                    df_std=global_covariates_std,
                                    df_min=global_covariates_min,
                                    df_max=global_covariates_max)

        new_data_tuple = (local_train_values.values.tolist(),
                          missing_values,
                          time_delta,
                          local_covariate.values.tolist()[0]    ,
                          local_histogram.values.tolist(),
                          y_labels)
        new_data_list.append((month_day_hour_key, new_data_tuple))

    return new_data_list


def evaluate_df_statistic(df: pd.DataFrame):
    df_mean, df_std, df_min, df_max = df.mean(), df.std(), df.min(), df.max()
    return df_mean, df_std, df_min, df_max
