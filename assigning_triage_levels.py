import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve



def f(row):
    if row['flag'] <= .3:
        val = 'G'
    elif .3 < row['flag'] <= .7:
        val = 'Y'
    elif .7 < row['flag'] <= 1:
        val = 'R'
    return val

def create_flag_column(model, X_test):
    """Creates a new dataframe of X_test combined with the green, yellow,
    and red, triage flags."""

    prediction_probabilities = model.predict_proba(X_test)[:, 1]
    pred_proba_df = pd.DataFrame({'pred_proba': prediction_probabilities})

    #add the predicted probabilities to the X_test dataset
    X_test_proba_appended = pd.concat((X_test.reset_index(), pred_proba_df), axis=1)

    # for the visualization, creating 'patient_id' - hypothetical patient ID numbers.
    X_test_proba_appended = X_test_proba_appended.rename(columns={'index':'patient_id'})

    # Adding an additional predict_proba column that will be turned into triage flag indicators
    X_test_proba_appended['flag'] = X_test_proba_appended.pred_proba

    X_test_proba_appended['flag'] = X_test_proba_appended.apply(f, axis=1)

    X_test_proba_appended.to_csv('prediction_with_flag.csv')

    return X_test_proba_appended

def add_flags_to_original_dataset(model, X_test, original_data):
    """Adds flags back to the original dataset without feature engineered
    columns for better visualization."""
    X_test_indices = X_test.index
    X_test_indices = [ind for ind in X_test_indices]
    X_test_from_original = original_data.loc[X_test_indices,:]
    X_test_from_original_flagged = pd.concat((X_test_from_original.reset_index(), pred_proba_df), axis=1)
    X_test_from_original_flagged = X_test_from_original_flagged.rename(columns={'index':'patient_id'})

    X_test_from_original_flagged['flag'] = X_test_from_original_flagged.pred_proba
    X_test_from_original_flagged['flag'] = X_test_from_original_flagged.apply(f, axis=1)

    X_test_proba_appended.to_csv('original_data_with_flag.csv')

    return X_test_proba_appended
