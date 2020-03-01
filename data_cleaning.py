import pandas as pd

def drop_null(data):
    """Drops any columns with > 50% of the columns missing. From remaining columsn,
    drops all rows with null values."""

    cols_with_many_null = (data.isna().sum() / len(data)).sort_values(ascending = False)
    cols_with_many_null = cols_with_many_null.index[cols_with_many_null > 0.50]
    print('There are %d columns with more than 50%% missing values' % len(cols_with_many_null))

    #drop columns with > 50% null values
    data = data.drop(cols_with_many_null, axis=1)
    print("Columns dropped")

    data = data.dropna()
    print("Rows with nulls dropped.")

    return data


def clean_data(data):
    #remove values of icu days, hospital death prob, and icu death prob that are less than 0
    #as these values are not possible
    data1 = data[data.pre_icu_los_days >= 0]
    data2 = data[data.apache_4a_hospital_death_prob >= 0]
    data3 = data[data.apache_4a_icu_death_prob >= 0]

    #fill in columns with many nulls to indicate weather test was performed or not
    data3.h1_lactate_max = data3.h1_lactate_max.fillna(0)
    data3.h1_lactate_max = data3.h1_lactate_max.apply(lambda x: 1 if x != 0 else x)
    data3.h1_inr_max = data3.h1_inr_max.fillna(0)
    data3.h1_inr_max = data3.h1_inr_max.apply(lambda x: 1 if x != 0 else x)
    data3.bilirubin_apache = data3.bilirubin_apache.fillna(0)
    data3.bilirubin_apache = data3.bilirubin_apache.apply(lambda x: 1 if x != 0 else x)

    #replace nulls in hospital_admit_source with 'unkown'
    data3.hospital_admit_source = data3.hospital_admit_source.fillna('unkown')

    #remove additional nulls from dataset
    cleaned_data = drop_null(data3)

    return cleaned_data
