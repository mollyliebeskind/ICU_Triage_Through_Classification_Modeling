"""
Deploys random forest classifier model to dataset to predict survival
probability in the ICU.

Model details
  -- 80% train and 20% test sets
  -- One hot encoded categorical features
  -- Random oversampling to account for imbalanced class sizes
"""

import pickle
import pandas as pd

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def dummy_data(data):
    """Returns a dataframe with one hot encoded categorical columns."""

    cat_cols = [col for col in data.columns if data[col].dtype == 'O']
    dummied_data = pd.get_dummies(data, drop_first=True, columns=cat_cols)

    return dummied_data

def create_x_y(data):
    """Returns X and y dataframes where y contains only the dependent variable,
    'hospital_death', and X contains all other columns.
    """

    X, y = data.drop('hospital_death', axis=1), data['hospital_death']
    return X, y


def tts(data):
    """Returns train and test dataframes by calling dummy_data and create_x_y
    and using a stratified train_test_split with a test_size of .2.
    """

    dummied_data = dummy_data(data)
    X, y = create_x_y(dummied_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


def final_model(data):
    """Splits data into train and test sets and deplys a random forest
    classifier to predict survival probability. Prints accuracy scores
    including recall and precision. Pickles the model.
    """

    X_train, X_test, y_train, y_test = tts(data)

    ros = RandomOverSampler(random_state=19)
    X_train, y_train = ros.fit_sample(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100,
                                   max_depth=5,
                                   min_samples_leaf=100)
    model.fit(X_train, y_train)
    prediction = prediction = model.predict(X_test)

    print("Model Recall: ", recall_score(y_test, prediction, average='binary', pos_label=1))
    print("Model Precision: ", precision_score(y_test, prediction))
    print("Model Accuracy: ", accuracy_score(y_test, prediction))

    pickle.dump(model, open('icu_survival_model.pkl', 'wb'))

    return

final_model()
