import numpy as np
import pandas as pd
from src.logger import logging
from data_ingestion import save_data

def negative_value_treatment(train: pd.DataFrame, test: pd.DataFrame):
    try:
        train.loc[train['Weekly_Sales'] < 0 ,'Weekly_Sales'] = 0
        train.loc[train['MarkDown2'] < 0 ,'MarkDown2'] = 0
        train.loc[train['MarkDown3'] < 0, 'MarkDown3'] = 0
        test.loc[test['MarkDown1'] < 0 ,'MarkDown1']= 0
        test.loc[test['MarkDown2'] < 0, 'MarkDown2'] = 0
        test.loc[test['MarkDown3'] < 0 ,'MarkDown3']= 0
        test.loc[test['MarkDown5'] < 0 ,'MarkDown5']= 0
    except Exception as e:
        logging.error("Error while treating negative values: %s", e)
        raise

def treating_nan(train: pd.DataFrame, test: pd.DataFrame):
    try:
        test['CPI'] = test.groupby(['Dept'])['CPI'].transform(lambda x: x.fillna(x.mean()))
        test['Unemployment'] = test.groupby(['Dept'])['Unemployment'].transform(lambda x: x.fillna(x.mean()))

        train = train.fillna(0)
        test = test.fillna(0)
    except Exception as e:
        logging.error("Error while treating NAN values %s", e)
        raise

def outlier_treatment(train: pd.DataFrame):
    try:
        train['Weekly_Sales'] = np.where(train['Weekly_Sales']>100000,100000,train['Weekly_Sales'])
    except Exception as e:
        logging.error("Error while treating outliers: %s", e)
        raise

def main():
    try:
        train = pd.read_csv('data/raw/merged_train.csv')
        test = pd.read_csv('data/raw/merged_test.csv')

        negative_value_treatment(train, test)
        treating_nan(train, test)
        outlier_treatment(train)

        save_data(train, test, data_path='./data', train_filename='processed_train', test_filename='processed_test', folder='processed')
    except Exception as e:
        logging.error("Error while data preprocessing %s", e)
        raise

if __name__=="__main__":
    main()
