import numpy as np
import pandas as pd
from src.logger import logging
from src.data.data_ingestion import save_data

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a csv file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def temperature(train: pd.DataFrame, test: pd.DataFrame):
    train['Temperature'] = (train['Temperature'] - 32) * 5/9
    test['Temperature'] = (test['Temperature'] - 32) * 5/9

def conv_to_date(train: pd.DataFrame, test: pd.DataFrame):
    train['Date'] = pd.to_datetime(train['Date'])
    test['Date'] = pd.to_datetime(test['Date'])

def extract_datetime(train: pd.DataFrame, test: pd.DataFrame):
    try:
        train["Date"]=pd.to_datetime(train["Date"])
        train["Day"]=train['Date'].dt.day
        train["Week"]=train['Date'].dt.week
        train["Month"]=train['Date'].dt.month
        train["Year"]=train['Date'].dt.year

        test["Day"]=test['Date'].dt.day
        test["Week"]=test['Date'].dt.week
        test["Month"]=test['Date'].dt.month
        test["Year"]=test['Date'].dt.year
    except Exception as e:
        logging.error("Error duing datetime extract %s", e)

def creating_holidays(train: pd.DataFrame, test: pd.DataFrame):
    try:
        s_1=np.datetime64('2010-02-12')
        s_2=np.datetime64('2011-02-11')
        s_3=np.datetime64('2012-02-10')
        s_4=np.datetime64('2013-02-08')

        #labor day
        l_1=np.datetime64('2010-09-10')
        l_2=np.datetime64('2011-09-09')
        l_3=np.datetime64('2012-09-07')
        l_4=np.datetime64('2013-09-06')

        #thanksgiving
        t_1=np.datetime64('2010-11-26')
        t_2=np.datetime64('2011-11-25')
        t_3=np.datetime64('2012-11-23')
        t_4=np.datetime64('2013-11-29')

        #christams day
        c_1=np.datetime64('2010-12-31')
        c_2=np.datetime64('2011-12-30')
        c_3=np.datetime64('2012-12-28')
        c_4=np.datetime64('2013-12-27')

        train['Superbowl']=np.where(((train.Date==s_1) | (train.Date==s_2) |
                                    (train.Date==s_3) | (train.Date==s_4)),1,0)

        train['labor']= np.where(((train.Date==l_1) | (train.Date==l_2) |
                                        (train.Date==l_3) | (train.Date==l_4)),1,0)

        train['thanksgiving']=np.where(((train.Date==t_1) | (train.Date==t_2) |
                                        (train.Date==t_3) | (train.Date==t_4)),1,0)

        train['christmas']=np.where(((train.Date==c_1) | (train.Date==c_2) |
                                    (train.Date==c_3) | (train.Date==c_4)),1,0)

        test['Superbowl']=np.where(((test.Date==s_1) | (test.Date==s_2) |
                                    (test.Date==s_3) | (test.Date==s_4)),1,0)

        test['labor']= np.where(((test.Date==l_1) | (test.Date==l_2) |
                                        (test.Date==l_3) | (test.Date==l_4)),1,0)

        test['thanksgiving']=np.where(((test.Date==t_1) | (test.Date==t_2) |
                                        (test.Date==t_3) | (test.Date==t_4)),1,0)

        test['christmas']=np.where(((test.Date==c_1) | (test.Date==c_2) |
                                    (test.Date==c_3) | (test.Date==c_4)),1,0)

        train['IsHoliday']=np.where((train.IsHoliday==True),1,0)

        test['IsHoliday']=np.where((test.IsHoliday==True),1,0)

        train['IsHoliday']=train['IsHoliday']|train['Superbowl']|train['labor']|train['thanksgiving']|train['christmas']
        
        test['IsHoliday']=test['IsHoliday']|test['Superbowl']|test['labor']|test['thanksgiving']|test['christmas']

        dplist=['Superbowl', 'labor', 'thanksgiving', 'christmas']
        train.drop(dplist,inplace=True,axis=1)
        test.drop(dplist,inplace=True,axis=1)

        train=pd.get_dummies(train, drop_first=True)
        test=pd.get_dummies(test,drop_first=True)
    except Exception as e:
        logging.error("Error during creating holidays %s", e)

def final_touch(train: pd.DataFrame, test: pd.DataFrame):
    try:
        dpcol=['MarkDown1', 'MarkDown5', 'Year' , 'Day', 'Month' , 'CPI', 'Unemployment'] 
        train.drop(dpcol,inplace=True,axis=1)
        test.drop(dpcol,inplace=True,axis=1)

        for var in train:
            if train[var].dtypes == float:
                train[var]=train[var].astype(int)
                
        for var in test:
            if test[var].dtypes == float:
                test[var]=test[var].astype(int)
    except Exception as e:
        logging.error("Error during final touch %s", e)
        raise

def main():
    train = load_data('data\processed\processed_train.csv')
    test = load_data('data\processed\processed_test.csv')

    temperature(train, test)
    conv_to_date(train, test)
    extract_datetime(train,test)
    creating_holidays(train, test)
    final_touch(train, test)

    save_data(train, test, data_path='./data', train_filename='train.csv', test_filename='test.csv', folder='interim')

if __name__=="__main__":
    main()