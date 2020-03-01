import seaborn as sns
import pandas as pd

def exploring_y(data):
    """Reveals a few key characterstics of the y_value, hospital_death."""
    print("Unique hospital death values:\n", data.hospital_death.unique())
    print("\nNumber of each y value:\n", data.hospital_death.value_counts())
    print("\nPercentage of survivors:\n", data.hospital_death.value_counts(normalize=True)*100)
    return

def category_observations(data, category, data_dict):
    """Supports create_cat_df by identifying columns that are in the
    given category. Categories are obtained using the data_dict file."""

    print(f"Number of columns in {category}: {data_dict[data_dict.Category == category].shape[0]}")
    category_col = data_dict[data_dict.Category == category].reset_index()
    cat_cols = [x for x in category_col['Variable Name'].unique() if x in data.columns]
    return cat_cols

def create_cat_df(data, category, data_dict, demo=False):
    """Takes in the dataset and category then calls category_observations
    to identify which columns belong in that category. Returns a dataframe
    with only those columns included."""

    if demo == True:
        selecting_cols = category_observations(data, category, data_dict)
    else:
        selecting_cols = category_observations(data, category, data_dict) + ['hospital_death']

    cat_only_data = data[selecting_cols]
    return cat_only_data

def pairplotting(data):
    """Creates a pairplot of the data"""

    sns.pairplot(data, plot_kws={'alpha':.2}, hue='hospital_death')
    return

def selecting_cols(data):
    """Takes in a dataframe and calculates the difference in mean between
    those that survive and do not survive for each column. Returns a dataframe
    of only columns with signfificant differentiation between the means as well
    as a list of the columns included in the Dataframe. Differentiation is
    considered signficant if mean(1)/mean(0) > 1.5 or <.75."""

    selected_cols = []
    dif_in_means = data.groupby('hospital_death').mean().iloc[0,:] / data.groupby('hospital_death').mean().iloc[1,:]
    high_end = dif_in_means.index[dif_in_means > 1.5].to_list()
    low_end = dif_in_means.index[dif_in_means < .75].to_list()
    selected_cols += high_end
    selected_cols += low_end
    selected_cols.append('hospital_death')
    if len(selected_cols) > 1:
        selected_data = data[selected_cols]
        return selected_data, selected_cols
    else:
        print("No signficant features. Additional exploration needed.")
        return dif_in_means, 0
