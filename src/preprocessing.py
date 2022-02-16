import numpy as np
import pandas as pd

import settings


class TabularPreprocessor:

    def __init__(self, df_train, df_test, shift_imputation=None):

        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.shift_imputation = shift_imputation

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
            df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week.astype(np.int64)
            df['DayOfWeek'] = df['DateTime'].dt.dayofweek
            df['DayOfYear'] = df['DateTime'].dt.dayofyear

    def create_aggregation_features(self):

        """
        Create aggregation features
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        continuous_features = ['AirTemperature']
        categorical_features = ['Year']

        for categorical_feature in categorical_features:
            for continuous_feaure in continuous_features:
                for aggregation in ['mean', 'std', 'min', 'max']:
                    df_agg = df_all.groupby(categorical_feature)[continuous_feaure].agg(aggregation)
                    self.df_train[f'{categorical_feature}_{continuous_feaure}_{aggregation}'] = self.df_train[categorical_feature].map(df_agg)
                    self.df_test[f'{categorical_feature}_{continuous_feaure}_{aggregation}'] = self.df_test[categorical_feature].map(df_agg)

    def create_lag_lead_features(self):

        """
        Create lag/lead features with imputation
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        continuous_features = ['AirTemperature', 'ComfortTemperature', 'RelativeHumidity', 'EffectiveCloudCover']

        if self.shift_imputation == 'aggregation':

            # Insert new datetime indices correspond to shifted periods
            datetimes_to_insert = [
                '2018-12-31 23:00:00',
                '2022-01-01 00:00:00', '2022-01-01 01:00:00', '2022-01-01 02:00:00',
                '2022-01-01 03:00:00', '2022-01-01 04:00:00', '2022-01-01 05:00:00',
            ]
            df_all = df_all.set_index('DateTime')
            for datetime in datetimes_to_insert:
                df_all = pd.concat((df_all, pd.DataFrame(index=[pd.to_datetime(datetime)])))
            df_all = df_all.reset_index().rename(columns={'index': 'DateTime'})

            # Use mean values of Month/DayOfMonth/HourOfDay group's to fill missing values
            df_all['Month'] = df_all['DateTime'].dt.month
            df_all['DayOfMonth'] = df_all['DateTime'].dt.day
            df_all['HourOfDay'] = df_all['DateTime'].dt.hour

            for continuous_feature in continuous_features:
                df_all[continuous_feature] = df_all.groupby(['Month', 'DayOfMonth', 'HourOfDay'])[continuous_feature].apply(lambda x: x.fillna(x.mean()))

        # Shift with imputed values
        for continuous_feature in continuous_features:
            for period in [-6, -5, -4, -3, -2, -1, 1]:

                if continuous_feature != 'EffectiveCloudCover' and period < -1:
                    continue

                if period > 0:
                    shifted_feature = df_all[continuous_feature].shift(periods=period).bfill().values
                else:
                    shifted_feature = df_all[continuous_feature].shift(periods=period).ffill().values

                self.df_train[f'{continuous_feature}_shift{period}'] = shifted_feature[0:len(self.df_train)]
                self.df_test[f'{continuous_feature}_shift{period}'] = shifted_feature[len(self.df_train):len(self.df_train) + len(self.df_test)]

    def create_rolling_features(self):

        """
        Create rolling features with imputation
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        continuous_features = ['AirTemperature', 'ComfortTemperature', 'RelativeHumidity', 'WindSpeed', 'WindDirection', 'EffectiveCloudCover']
        for continuous_feature in continuous_features:
            for window in [3, 6]:
                for aggregation in ['mean', 'std']:
                    rolling_feature = df_all.rolling(window=window, min_periods=1)[continuous_feature].agg(aggregation).values
                    self.df_train[f'{continuous_feature}_rolling{window}_{aggregation}'] = rolling_feature[0:len(self.df_train)]
                    self.df_test[f'{continuous_feature}_rolling{window}_{aggregation}'] = rolling_feature[len(self.df_train):]

    def create_sun_features(self):

        """
        Read and merge pre-computed sun features
        """

        df_sun = pd.read_csv(settings.DATA / 'sun.csv')
        df_sun['Date'] = pd.to_datetime(df_sun['Date'])
        df_sun['Dawn'] = pd.to_datetime(df_sun['Dawn']).dt.time
        df_sun['Sunrise'] = pd.to_datetime(df_sun['Sunrise']).dt.time
        df_sun['Noon'] = pd.to_datetime(df_sun['Noon']).dt.time
        df_sun['Sunset'] = pd.to_datetime(df_sun['Sunset']).dt.time
        df_sun['Dusk'] = pd.to_datetime(df_sun['Dusk']).dt.time
        self.df_train = self.df_train.merge(df_sun, how='left', on='Date')
        self.df_test = self.df_test.merge(df_sun, how='left', on='Date')

        for df in [self.df_train, self.df_test]:
            df['IsBetweenSunriseAndSunset'] = np.int64((df['Time'] > df['Sunrise']) & (df['Time'] < df['Sunset']))
            df['IsBetweenDawnAndDusk'] = np.int64((df['Time'] > df['Dawn']) & (df['Time'] < df['Dusk']))

    def transform(self):

        self.get_folds()
        self.create_datetime_features()
        self.create_aggregation_features()
        self.create_lag_lead_features()
        self.create_rolling_features()
        self.create_sun_features()

        return self.df_train, self.df_test
