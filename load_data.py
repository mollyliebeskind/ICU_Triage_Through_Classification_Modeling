import pandas as pd
import numpy as np
import psycopg2 as pg
import pandas as pd
import pandas.io.sql as pd_sql
from sqlalchemy import create_engine

def access_aws(host, database, query):
    """Connects to AWS host and database. Queries data based on query
    input. Returns a Pandas dataframe with querried data."""

    connection_args = {
    'host': host, #replace with instance ip address
    'dbname': database,
    'user': 'mollyliebeskind',
    'port': 5432
}
    connection = pg.connect(**connection_args)

    from sqlalchemy import create_engine
    connection_string = f'postgres://ubuntu:{connection_args["user"]}@{connection_args["host"]}:{connection_args["port"]}/{connection_args["dbname"]}'
    engine = create_engine(connection_string)
    print("Connected to AWS. Querying database and creating dataframe.")

    cursor = connection.cursor()
    query = query
    df = pd.read_sql(query,engine)
    print("Dataframe created.")

    return df

def import_files(file_name):
    """Reads in CSV files and converts them to dataframes"""
    imported_file = pd.read_csv(f'{file_name}')
    print(f"{file_name} was imported")
    return imported_file

def basic_data_info(data):
    """Provides basic information including number of rows, number of columns,
    number of unique patient ids, encounters, hospital ids, and icu ids."""

    print("\nTotal number of instances:", data.shape[0])
    print("Total number of columns", data.shape[1])
    print("Unique patient_ids:", data.patient_id.nunique())
    print("Unique encounter_ids:", data.encounter_id.nunique())
    print("Unique hospital_ids:", data.hospital_id.nunique())
    print("Unique icu_ids:", data.icu_id.nunique())

    return

def load_and_info(data):
    raw_data = import_files(data)
    basic_data_info(raw_data)
    return raw_data
