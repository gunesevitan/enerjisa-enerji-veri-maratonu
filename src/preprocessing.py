import numpy as np
import pandas as pd

import settings


class TabularPreprocessor:

    def __init__(self, df_train, df_test, fill_missing_values=False):

        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.fill_missing_values = fill_missing_values

    def get_folds(self):

        """
        Read and merge pre-computed folds
        """

        df_folds = pd.read_pickle(settings.DATA / 'folds.pkl')
        self.df_train = self.df_train.merge(df_folds, how='left', on='DateTime')

    def clean_features(self):

        """
        Clean features
        """

        # Change WWCode 84 (Shower(s) of rain and snow, moderate or heavy) to 83 (Shower(s) of rain and snow, slight)
        # WWCode 84 doesn't exist in training set and the most similar category is 83
        self.df_test.loc[self.df_test['WWCode'] == 84, 'WWCode'] = 83

        if self.fill_missing_values:
            self.df_train['WWCode'] = pd.read_csv(settings.DATA / 'wwcode_filled.csv')['WWCode_Filled']

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
            for continuous_feature in continuous_features:
                for aggregation in ['mean', 'std', 'min', 'max']:
                    df_agg = df_all.groupby(categorical_feature)[continuous_feature].agg(aggregation)
                    self.df_train[f'{categorical_feature}_{continuous_feature}_{aggregation}'] = self.df_train[categorical_feature].map(df_agg)
                    self.df_test[f'{categorical_feature}_{continuous_feature}_{aggregation}'] = self.df_test[categorical_feature].map(df_agg)

    def create_lag_lead_features(self):

        """
        Create lag/lead features with or without imputation
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        continuous_features = ['AirTemperature', 'ComfortTemperature', 'RelativeHumidity', 'EffectiveCloudCover']

        # Shift features with or without imputation
        for continuous_feature in continuous_features:

            for period in [-6, -5, -4, -3, -2, -1, 1]:

                if period > 0:
                    if self.fill_missing_values:
                        shift_feature = df_all[continuous_feature].shift(periods=period).bfill().values
                    else:
                        shift_feature = df_all[continuous_feature].shift(periods=period).values
                else:
                    if self.fill_missing_values:
                        shift_feature = df_all[continuous_feature].shift(periods=period).ffill().values
                    else:
                        shift_feature = df_all[continuous_feature].shift(periods=period).values

                self.df_train[f'{continuous_feature}_shift{period}'] = shift_feature[:len(self.df_train)]
                self.df_test[f'{continuous_feature}_shift{period}'] = shift_feature[len(self.df_train):]

            # Diff features with or without imputation
            for period in [1]:

                if self.fill_missing_values:
                    diff_feature = df_all[continuous_feature].diff(periods=period).fillna(0).values
                else:
                    diff_feature = df_all[continuous_feature].diff(periods=period).values

                self.df_train[f'{continuous_feature}_diff{period}'] = diff_feature[:len(self.df_train)]
                self.df_test[f'{continuous_feature}_diff{period}'] = diff_feature[len(self.df_train):]

            # Percentage change features with or without imputation
            for period in [-1, 1]:

                if self.fill_missing_values:
                    pct_change_feature = df_all[continuous_feature].pct_change(periods=period).fillna(0).values
                else:
                    pct_change_feature = df_all[continuous_feature].pct_change(periods=period).values

                self.df_train[f'{continuous_feature}_pct_change{period}'] = pct_change_feature[:len(self.df_train)]
                self.df_test[f'{continuous_feature}_pct_change{period}'] = pct_change_feature[len(self.df_train):]

    def create_rolling_features(self):

        """
        Create rolling features with imputation
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        continuous_features = ['AirTemperature', 'ComfortTemperature', 'RelativeHumidity', 'EffectiveCloudCover']

        for continuous_feature in continuous_features:
            for window in [3, 6]:
                for aggregation in ['mean', 'std']:
                    rolling_feature = df_all.rolling(window=window, min_periods=1)[continuous_feature].agg(aggregation).values
                    self.df_train[f'{continuous_feature}_rolling{window}_{aggregation}'] = rolling_feature[:len(self.df_train)]
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
        self.clean_features()
        self.create_datetime_features()
        self.create_aggregation_features()
        self.create_lag_lead_features()
        self.create_rolling_features()
        self.create_sun_features()

        return self.df_train, self.df_test
