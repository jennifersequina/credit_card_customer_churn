import pandas as pd
import numpy as np


class WrongMetricError(Exception):
    pass

def _accuracy_validation(new_y: pd.DataFrame) -> float:

    # check the percentage of matched predictions and real value
    new_y['valid'] = np.where(new_y['predictions'] == new_y['real_value'], 1, 0)
    count_valid = new_y['valid'].sum().astype(float)
    total_count = new_y['valid'].count().astype(float)
    result = np.round(count_valid / total_count, 2)
    return result


def _precision_validation(new_y: pd.DataFrame) -> float:

    # check the number of true positive and false positve
    true_positive = ((new_y['predictions'] == 1) & (new_y['real_value'] == 1)).sum().astype(float)
    false_positive = ((new_y['predictions'] == 1) & (new_y['real_value'] == 0)).sum().astype(float)
    result = np.round(true_positive / (true_positive + false_positive), 2)
    return result


def _recall_validation(new_y: pd.DataFrame) -> float:

    # check the number of true positive and false negative
    true_positive = ((new_y['predictions'] == 1) & (new_y['real_value'] == 1)).sum().astype(float)
    false_negative = ((new_y['predictions'] == 0) & (new_y['real_value'] == 1)).sum().astype(float)
    result = np.round(true_positive / (true_positive + false_negative), 2)
    return result


def _f1_validation(new_y: pd.DataFrame) -> float:
    # get precision score
    precision_score = _precision_validation(new_y)

    # get recall score
    recall_score = _recall_validation(new_y)

    # compute the f_score
    result = np.round((2 * (recall_score * precision_score)) / (recall_score + precision_score), 2).astype(float)
    return result

def _confusion_validation(new_y: pd.DataFrame) -> None:
    true_positive = ((new_y['predictions'] == 1) & (new_y['real_value'] == 1)).sum().astype(float)
    false_positive = ((new_y['predictions'] == 0) & (new_y['real_value'] == 1)).sum().astype(float)
    false_negative = ((new_y['predictions'] == 1) & (new_y['real_value'] == 0)).sum().astype(float)
    true_negative = ((new_y['predictions'] == 0) & (new_y['real_value'] == 0)).sum().astype(float)
    print(f'True Positive is {true_positive}, False Positive is {false_positive}')
    print(f'False Negative is {false_negative}, True Negative is {true_negative}')


def validate(test_df: pd.DataFrame,
             test_labels: pd.Series,
             ml_model,
             metric_name: str,
             confusion_matrix: bool) -> float:

    metrics_dict = dict()
    metrics_dict['accuracy'] = _accuracy_validation
    metrics_dict['precision'] = _precision_validation
    metrics_dict['recall'] = _recall_validation
    metrics_dict['f1'] = _f1_validation

    new_y = test_labels.to_frame(name='real_value').reset_index().rename(columns={'index': 'ID'})
    new_y['predictions'] = ml_model.predict(test_df)

    try:
        score = metrics_dict[metric_name](new_y)
        print(f"{metric_name.capitalize()} equals to {score} for model {ml_model}")
        if confusion_matrix:
            _confusion_validation(new_y)
        return score

    except KeyError as e:
        raise WrongMetricError(f'''Metric name {metric_name} is not correct, 
        please provide one of accuracy, precision, recall, f1''')

