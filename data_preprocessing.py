"""
This script loads the icu mortality dataset from PostgreSQL in AWS EC2
environment. Null values are then removed or imputed and feature engineering
is performed so that only columns that showed predictive power in exploratory
data analysis are maintained. The final dataset is saved as a csv file.
"""

import pandas as pd

from connect_to_aws import access_postgres_in_aws

# Load the data
def access_aws(data_query):
    """Returns a dataframe with querried information. Connects to
    PostgreSQL in AWS EC2 environment.

    Args:
    data_query -- a SQL query
    """

    # connects to database in postgreSQL - info stored in separate doc
    engine = access_postgres_in_aws()
    query = data_query
    icu_dataframe = pd.read_sql(query, engine)

    return icu_dataframe

def import_data():
    """Returns a dataframe with all ICU data and a dataframe of the
    data dictionary.
    """
    icu_data = access_aws('SELECT * FROM icu_mortality_table')
    data_dict = access_aws('SELECT * FROM icu_data_dict_table')

    return icu_data, data_dict

# Data Cleaning
def drop_null(data):
    """Returns a dataframe without null values. Any columns that have > 50% of
    null values is removed.
    """

    # Identify which columns to drop
    cols_with_many_null = (data.isna().sum()/len(data)).sort_values(ascending
                                                                    =False)
    cols_with_many_null = cols_with_many_null.index[cols_with_many_null > 0.50]

    # Drop columns with > 50% null values
    data = data.drop(cols_with_many_null, axis=1)

    return data

def clean_data(data):
    """Returns a dataframe with impossible values removed and null values
    removed.

    The following columns cannot have values < 0 as they are durations and
    probabilities.
      -- pre_icu_los_days
      -- apache_4a_hospital_death_prob
      -- apache_4a_icu_death_prob

    Nulls in hospital_admit_source are replaced with 'unkown' and all other
    null values are dropped.
    """

    # Drop impossible values
    data = data[data.pre_icu_los_days >= 0]
    data = data[data.apache_4a_hospital_death_prob >= 0]
    data = data[data.apache_4a_icu_death_prob >= 0]


    # Handle null values
    data.hospital_admit_source = data.hospital_admit_source.fillna('unkown')
    data = drop_null(data)

    return data

# Feature Engineering
def category_observations(data, category, data_dict):
    """Returns all columns in a given category as indicated by the data
    dictionary.

    Args:
    data_dict -- the ICU data dictionary
    category -- the category which all column names should be returned for
    """

    category_col = data_dict[data_dict.Category == category].reset_index()
    cat_cols = [x for x in category_col['Variable Name'].unique()
                if x in data.columns]
    return cat_cols

def feature_selection(data, data_dict):
    """Returns a dataframe with only the deomographic, apache covariate,
    and apache prediction features. These features showed the most predictive
    power in explorator data analysis.

    Args:
    data -- cleaned dataframe
    data_dict -- the ICU data dictionary
    """

    demo_cols = category_observations(data, 'demographic', data_dict)
    apach_cov_cols = category_observations(data, 'APACHE covariate',
                                           data_dict)
    apach_pred_cols = category_observations(data, 'APACHE prediction',
                                            data_dict)
    features = demo_cols + apach_cov_cols + apach_pred_cols
    selected_df = data[features]

    return selected_df

def feature_engineering(data):
    """Returns a dataframe with the pre_icu_los_days and the apache features
    amplified to increase signal and reduce noise.
    """

    data.pre_icu_los_days = data.pre_icu_los_days.apply(lambda x: x**2)
    data['bun_creatinine'] = data.bun_apache * data.creatinine_apache
    data = data.drop(['bun_apache', 'creatinine_apache'], axis=1)

    return data

def main():
    """Loads the ICU dataset, performs data cleaning and feature engineering,
    and saves the dataset as a csv file.
    """
    # Loading data
    icu_data, data_dict_df = import_data()

    # Cleaning data
    icu_data = drop_null(icu_data)
    icu_data = clean_data(icu_data)

    # Feature engineering
    model_df = feature_selection(icu_data, data_dict_df)
    model_df = feature_engineering(model_df)

    model_df.to_csv('model_dataframe.csv')

main()
