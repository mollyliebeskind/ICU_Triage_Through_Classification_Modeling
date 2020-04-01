"""
Adds flags indicating survival risk to the X_test and original dataframes.
Saves all dataframes as csv files for visualization in Tableau.

Files saved include
-- X_test_with_flag.csv
-- original_data_with_flag.csv
"""

import pickle
import pandas as pd

def flag(row):
    """Replaces any value in column 'flag' with flags to indicate prbability
    of survival where
      -- R = high risk of not surviving
      -- Y = medium risk of not surviving
      -- G = low risk of not surviving
    """
    if row['flag'] <= .3:
        val = 'G'
    elif .3 < row['flag'] <= .7:
        val = 'Y'
    elif .7 < row['flag'] <= 1:
        val = 'R'
    return val

def create_flag_column(x_df, saved_model):
    """Adds flags to the X_test dataframe that indicate survival risk level
    where
      -- R = high risk of not surviving
      -- Y = medium risk of not surviving
      -- G = low risk of not surviving

    Args:
    x_df -- the X_test dataframe created in modeling
    saved_model -- the model used to predict surivival probability
    """

    pred_probas = saved_model.predict_proba(x_df)[:, 1]
    pred_proba_df = pd.DataFrame({'pred_proba': pred_probas})

    # Adding flags to indicate risk level. R = high risk, Y = medium, G = low
    X_test_probs = pd.concat((X_test.reset_index(), pred_proba_df), axis=1)
    X_test_probs['flag'] = X_test_probs.pred_proba.apply(flag, axis=1)

    # for the visualization, creating 'patient_id' - hypothetical patient ID numbers.
    X_test_probs = X_test_probs.rename(columns={'index':'patient_id'})

    # Save as csv for visualization in Tableau
    X_test_probs.to_csv('X_test_with_flag.csv')

    return X_test_probs


def flag_original(X_test, original_data, predict_proba_df):
    """Saves the original dataframe with flags appended to any row that was
    included in the X_test set. Dataframe will be used for Tableau
    visualization.
    """

    # Identify the X_test rows within the original dataset
    X_test_indices = list(X_test.index)
    filtered_original = original_data.loc[X_test_indices, :]
    original_flagged = pd.concat((filtered_original.reset_index(),
                                  predict_proba_df), axis=1)
    original_flagged = original_flagged.rename(columns={'index':'patient_id'})

    # Add R, G, Y flags to dataframe
    original_flagged['flag'] = original_flagged.pred_proba.apply(flag, axis=1)

    # Save as csv for Tableau visualization
    original_flagged.to_csv('original_data_with_flag.csv')

def main():
    """Loads necessary files for appending risk flags. Calls functions to
    add survival risk flags to the X_test and original datasets.
    """
    # Load necessary files
    X_test = pd.read_csv('X_test.csv')
    original_dataset = pd.read_csv('original_dataset.csv')
    model = pickle.load(open('icu_survival_model.pkl', 'rb'))

    # Append flags to X_test
    test_pred_proba = create_flag_column(X_test, model)

    # Append flags to original dataset
    flag_original(X_test, original_dataset, test_pred_proba)

main()
