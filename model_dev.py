import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, VotingClassifier,
                              AdaBoostClassifier, BaggingRegressor)

from ipywidgets import interactive, FloatSlider

def dummy_data(data):
    """Creates dummy variables for each categorical column."""

    cat_cols = [col for col in data.columns if (data[col].dtype == 'O')]
    dummied_data = pd.get_dummies(data, drop_first=True, columns=cat_cols)
    print("Dummy variables created.")
    return dummied_data

def create_x_y(data):
    """Creates X and y variables where y = 'hospital_death' column and
    X = remaining columns."""

    X, y = data.drop('hospital_death', axis=1), data['hospital_death']
    return X, y


def tts(data):
    """Splits the dataset into train and test sets. Calls create_x_y to
    split X and y, Calls dummy_data to turn any categorical columns into
    dummy variables."""
    dummied_data = dummy_data(data)

    X, y = create_x_y(dummied_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, stratify = y)
    print("Dataset split into train and test sets.\n")
    print("X_train shape:", X_train.shape)
    print("X_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test

def roc_curve_plotting(fpr, tpr, auc_roc):
    """Takes in fpr, tpr, and auc_roc to produce an ROC curve for all
    validation sets."""

    plt.figure(figsize=(5,5))
    lw = 2
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i],
                 lw=lw, label='ROC curve (area = %0.2f)' % auc_roc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
    plt.show()
    return

def scaling_data(X_tr, X_val):
    """Separates train and validation datasets into categorical and numerical columns.
    Scales numerical columns using StandardScaler"""

    #separating numerical columns in train and validation sets
    num_cols = [col for col in X_tr.columns if (X_tr[col].dtype == float)]
    X_tr_num, X_tr_not_num = X_tr[num_cols], X_tr.drop(num_cols, axis=1)
    X_val_num, X_val_not_num = X_val[num_cols], X_val.drop(num_cols, axis=1)

    #scaling numerical columns only
    ss = StandardScaler()
    X_tr_scaled = ss.fit_transform(X_tr_num)
    X_val_scaled = ss.transform(X_val_num)

    #rejoining scaled columns with non-scaled columns
    X_tr = np.concatenate((X_tr_scaled, X_tr_not_num), axis=1)
    X_val = np.concatenate((X_val_scaled, X_val_not_num), axis=1)

    return X_tr, X_val


def test_model_framework(X_train, y_train, model, scaling=False, smote=False, rand=False):
    """Within the 80% train data, splits the model into 5 folds for cross-validation testing.
    If model being implemented requires scaling, allows scaling function to be called.
    Also enables smote or random oversampling to be called."""

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 71)
    print("Kfolds created.")

    recall, precision, accuracy, matrix, roc_auc = [], [], [], [], []
    fpr, tpr = [], []
    p, t, pr_thresholds = [], [], []

    for train_ind, val_ind in kf.split(X_train,y_train):

        X_tr, y_tr = X_train.iloc[train_ind], y_train.iloc[train_ind]
        X_val, y_val = X_train.iloc[val_ind], y_train.iloc[val_ind]

        #standard scaling
        if scaling:
            X_tr, X_val = scaling_data(X_tr, X_val)

        if smote:
            ros = SMOTE(random_state=19)
            X_tr, y_tr = ros.fit_sample(X_tr, y_tr)

        if rand:
            ros = RandomOverSampler(random_state=19)
            X_tr, y_tr = ros.fit_sample(X_tr, y_tr)

        mod = model
        mod.fit(X_tr, y_tr)
        prediction_prob = mod.predict_proba(X_val)
        prediction = mod.predict(X_val)

        recall.append(recall_score(y_val, prediction, average='binary', pos_label=1))
        precision.append(precision_score(y_val, prediction, average='binary', pos_label=1))
        accuracy.append(accuracy_score(y_val, prediction))
        matrix.append(confusion_matrix(y_val, prediction))

        fpr_ex, tpr_ex, thresholds_ex = roc_curve(y_val, prediction_prob[:,1], pos_label=1)
        fpr.append(fpr_ex)
        tpr.append(tpr_ex)
        roc_auc.append(roc_auc_score(y_val, prediction_prob[:,1]))

    print("Confusion matrix:\n", (matrix[0] + matrix[1] + matrix[2] + matrix[3] + matrix[4])/5)
    print("Mean recall: ", np.mean(recall))
    print("Mean precision: ", np.mean(precision))
    print("Mean accuracy: ", np.mean(accuracy))

    roc_curve_plotting(fpr, tpr, roc_auc)

    return mod

def forced_grid_search(X_train, y_train):
    for n in [100,200,500]:
        for m in [5,10,50,100,500]:
            for z in [5,10,50,100,500]:
                print(f"For n_estimators = {n}, max_depth = {m}, min_samples_leaf={z}")
                rf_tuning = RandomForestClassifier(n_estimators=n,
                                                max_depth=m,
                                                min_samples_leaf=10)

                tuning_model_run = test_model_framework(X_train, y_train, rf_tuning,
                                                scaling=False, smote=False, rand=True)

    return

def final_model(X_train, X_test, y_train, y_test):

    ros = RandomOverSampler(random_state=19)
    X_train, y_train = ros.fit_sample(X_train, y_train)

    model = RandomForestClassifier(n_estimators = 100,
                                   max_depth = 5,
                                   min_samples_leaf=100)
    model.fit(X_train, y_train)
    prediction = prediction = model.predict(X_test)

    print("Model Recall: ", recall_score(y_test, prediction, average='binary', pos_label=1))
    print("Model Precision: ", precision_score(y_test, prediction))
    print("Model Accuracy: ", accuracy_score(y_test, prediction))

    return model

def make_confusion_matrix(X_test, y_test, model, threshold):
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    icu_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=80)
    sns.heatmap(icu_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['Surviving', 'Not-surviving'],
           yticklabels=['Surviving', 'Not-surviving']);
    plt.xlabel('prediction')
    plt.ylabel('actual')
    recall = icu_confusion[1][1] / (icu_confusion[1][1] + icu_confusion[1][0])
    precision = icu_confusion[1][1] / (icu_confusion[1][1] + icu_confusion[0][1])
    print('Recall: ', recall)
    print('Precision: ', precision)
    th_prec = pd.DataFrame({'threshold': threshold, 'prediction': y_predict})

    return
