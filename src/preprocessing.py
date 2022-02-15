import pandas as pd

import settings


class Preprocessor:

    def __init__(self, df_train, df_test):

        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)

    def get_folds(self):

        """
        Read and merge pre-computed folds
        """

        df_folds = pd.read_pickle(settings.DATA / 'folds.pkl')
        self.df_train = self.df_train.merge(df_folds, how='left', on='DateTime')

    def create_datetime_features(self):

        """
        Create basic datetime features from DateTime column
        """

        for df in [self.df_train, self.df_test]:
            df['Date'] = pd.to_datetime(df['DateTime'].dt.date)
            df['Time'] = df['DateTime'].dt.time
            df['Year'] = df['DateTime'].dt.year
            df['Month'] = df['DateTime'].dt.month
            df['DayOfMonth'] = df['DateTime'].dt.day
            df['HourOfDay'] = df['DateTime'].dt.hour
            df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek
            df['DayOfYear'] = df['DateTime'].dt.dayofyear

    def transform(self):

        self.get_folds()
        self.create_datetime_features()

        return self.df_train, self.df_test
