import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging

def load_params(params_path: str) -> dict:
    """" load parameters from a yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Params retrived from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected Error %s', e)
        raise

def load_data(data_url:str) -> pd.DataFrame:
    """load data from a csv file."""
    try:
        df = pd.read_csv(data_url)
        logging.info('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the csv file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error while loading the data %s', e)
        raise

def  Missing_Inspection(x):
    return pd.Series([x.count(),x.isnull().sum()],index = ['N',"NMISS"])

#UDF to Create Numerical Data Audit Report
def  num_var_summary(x):
    return pd.Series([x.count(),x.isnull().sum(),x.sum(),x.mean(),x.median(),x.std(),x.var(),x.min(),x.dropna().quantile(0.01),x.dropna().quantile(0.05),x.dropna().quantile(0.10),x.dropna().quantile(0.25),x.dropna().quantile(0.50),x.dropna().quantile(0.75),x.dropna().quantile(0.90),x.dropna().quantile(0.95),x.dropna().quantile(0.99),x.max()],
                    index = ['N',"NMISS","SUM",'MEAN','MEDIAN','STD','VAR','MIN','P1','P5','P10','P25','P50','P75','P90','P95','P99','MAX'])

# UDF to create categorical data audit report
def cat_var_summary(x):
    return pd.Series([x.count(),x.isnull().sum(),x.value_counts()], index=['N','NMISS','ColumnsName'])

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str, train_filename: str = 'train.csv', test_filename: str = 'test.csv', folder: str='raw') -> None:
    try:
        raw_data_path = os.path.join(data_path, folder)
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, train_filename), index=False)
        test_data.to_csv(os.path.join(raw_data_path, test_filename), index=False)
        logging.debug('Train and test data saved to %s as %s and %s', raw_data_path, train_filename, test_filename)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Load and merge data once
        train = pd.read_csv('data/external/train.csv')
        feature = pd.read_csv('data/external/features.csv')
        test = pd.read_csv('data/external/test.csv')
        stores = pd.read_csv('data/external/stores.csv')

        train = pd.merge(pd.merge(train, stores), feature, on=["Store", "Date", 'IsHoliday'])
        test = pd.merge(pd.merge(test, stores), feature, on=["Store", "Date", 'IsHoliday'])

        # Save raw merged data
        save_data(train, test, data_path='./data', train_filename='merged_train.csv', test_filename='merged_test.csv', folder='raw')

        # Get summaries
        num_summary_train = train.select_dtypes(include=['float64', 'float32','int32','int64']).apply(num_var_summary).T
        num_summary_test = test.select_dtypes(include=['float64', 'float32','int32','int64']).apply(num_var_summary).T

        cat_summary_train = train.select_dtypes(include=['object', 'O']).apply(cat_var_summary).T
        cat_summary_test = test.select_dtypes(include=['object', 'O']).apply(cat_var_summary).T

        # Save summaries
        save_data(num_summary_train, num_summary_test, './data', train_filename='num_summary_train.csv', test_filename='num_summary_test.csv')
        save_data(cat_summary_train, cat_summary_test, './data', train_filename='cat_summary_train.csv', test_filename='cat_summary_test.csv')
    except Exception as e:
        logging.error('Failed to complete data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__=='__main__':
    main()