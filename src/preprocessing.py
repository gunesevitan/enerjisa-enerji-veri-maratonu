import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import settings


class TabularPreprocessor:

    def __init__(self, df_train, df_test, fill_missing_values=False, normalize=False):

        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.fill_missing_values = fill_missing_values
        self.normalize = normalize

    def get_folds(self):

        """
        Read and merge pre-computed folds
        """

        df_folds = pd.read_pickle(settings.DATA / 'folds.pkl')
        self.df_train = self.df_train.merge(df_folds, how='left', on='DateTime')

    def clean_features(self):

        """
        Clean features and target
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
        continuous_features = ['AirTemperature', 'RelativeHumidity']
        categorical_features = ['Year']

        for categorical_feature in categorical_features:
            for continuous_feature in continuous_features:
                for aggregation in ['mean', 'std', 'min', 'max']:
                    df_agg = df_all.groupby(categorical_feature)[continuous_feature].agg(aggregation)
                    self.df_train[f'{categorical_feature}_{continuous_feature}_{aggregation}'] = self.df_train[categorical_feature].map(df_agg)
                    self.df_test[f'{categorical_feature}_{continuous_feature}_{aggregation}'] = self.df_test[categorical_feature].map(df_agg)

        for continuous_feature in continuous_features:
            for aggregation in ['mean', 'std', 'min', 'max']:
                df_agg = df_all.groupby(['Year', 'Month'])[continuous_feature].transform(aggregation).values
                self.df_train[f'Year_Month_{continuous_feature}_{aggregation}'] = df_agg[:len(self.df_train)]
                self.df_test[f'Year_Month_{continuous_feature}_{aggregation}'] = df_agg[len(self.df_train):]

    def create_lag_lead_features(self):

        """
        Create lag/lead features with or without imputation
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        continuous_features = ['AirTemperature', 'ComfortTemperature', 'RelativeHumidity', 'EffectiveCloudCover']

        # Shift features with or without imputation
        for continuous_feature in continuous_features:

            for period in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 1]:

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
            for period in [-1, 1]:

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
                for aggregation in ['mean', 'std', 'sum', 'var']:
                    rolling_feature = df_all.rolling(window=window, min_periods=1)[continuous_feature].agg(aggregation).fillna(0).values
                    self.df_train[f'{continuous_feature}_rolling{window}_{aggregation}'] = rolling_feature[:len(self.df_train)]
                    self.df_test[f'{continuous_feature}_rolling{window}_{aggregation}'] = rolling_feature[len(self.df_train):]

    def create_sun_features(self):

        """
        Read and merge pre-computed sun features
        """

        # Read sun features and convert them to datetime
        df_sun = pd.read_csv(settings.DATA / 'sun.csv')
        df_sun['Date'] = pd.to_datetime(df_sun['Date'])
        df_sun['Dawn'] = pd.to_datetime(df_sun['Dawn']).dt.time
        df_sun['Sunrise'] = pd.to_datetime(df_sun['Sunrise']).dt.time
        df_sun['Noon'] = pd.to_datetime(df_sun['Noon']).dt.time
        df_sun['Sunset'] = pd.to_datetime(df_sun['Sunset']).dt.time
        df_sun['Dusk'] = pd.to_datetime(df_sun['Dusk']).dt.time
        self.df_train = self.df_train.merge(df_sun, how='left', on='Date')
        self.df_test = self.df_test.merge(df_sun, how='left', on='Date')

        # Create binary day-night masks
        for df in [self.df_train, self.df_test]:
            df['IsBetweenSunriseAndSunset'] = np.int64((df['Time'] > df['Sunrise']) & (df['Time'] < df['Sunset']))
            df['IsBetweenDawnAndDusk'] = np.int64((df['Time'] > df['Dawn']) & (df['Time'] < df['Dusk']))

        # Create total day-time feature for every day
        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)
        for sun_feature in ['IsBetweenSunriseAndSunset', 'IsBetweenDawnAndDusk']:
            agg_feature = df_all.groupby(['Year', 'DayOfYear'])[sun_feature].transform('sum').values
            self.df_train[f'Year_DayOfYear_{sun_feature}_sum'] = agg_feature[:len(self.df_train)]
            self.df_test[f'Year_DayOfYear_{sun_feature}_sum'] = agg_feature[len(self.df_train):]

    def create_target_features(self):

        """
        Create target aggregation features
        """

        categorical_features = ['Month', 'WeekOfYear', 'DayOfYear']

        for categorical_feature in categorical_features:
            for aggregation in ['mean', 'std', 'min', 'max']:
                df_agg = self.df_train.groupby(categorical_feature)['Generation'].agg(aggregation)
                self.df_train[f'{categorical_feature}_Generation_{aggregation}'] = self.df_train[categorical_feature].map(df_agg)
                self.df_test[f'{categorical_feature}_Generation_{aggregation}'] = self.df_test[categorical_feature].map(df_agg)

    def normalize_features(self):

        """
        Normalize continuous features
        """

        df_all = pd.concat((self.df_train, self.df_test), axis=0, ignore_index=True)

        # Standardize continuous features
        continuous_features = [
            'AirTemperature', 'ComfortTemperature', 'RelativeHumidity', 'EffectiveCloudCover',
            'Year_AirTemperature_mean', 'Year_AirTemperature_std', 'Year_AirTemperature_min', 'Year_AirTemperature_max',
            'AirTemperature_shift-1', 'ComfortTemperature_shift-1', 'RelativeHumidity_shift-1',
            'EffectiveCloudCover_shift-1', 'EffectiveCloudCover_shift-2', 'EffectiveCloudCover_shift-3',
            'EffectiveCloudCover_shift-4', 'EffectiveCloudCover_shift-5', 'EffectiveCloudCover_shift-6',
            'EffectiveCloudCover_shift1', 'EffectiveCloudCover_diff1',
            'AirTemperature_rolling3_mean', 'AirTemperature_rolling3_std',
            'AirTemperature_rolling6_mean', 'AirTemperature_rolling6_std',
            'EffectiveCloudCover_rolling3_mean', 'EffectiveCloudCover_rolling3_std',
        ]
        scaler = StandardScaler()
        for continuous_feature in continuous_features:
            scaler.fit(df_all[continuous_feature].values.reshape(-1, 1))
            self.df_train[continuous_feature] = scaler.transform(self.df_train[continuous_feature].values.reshape(-1, 1))
            self.df_test[continuous_feature] = scaler.transform(self.df_test[continuous_feature].values.reshape(-1, 1))

        # Encode cyclical continuous features
        for df in [self.df_train, self.df_test]:
            df['MonthSin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['MonthCos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['HourOfDaySin'] = np.sin(2 * np.pi * df['HourOfDay'] / 24)
            df['HourOfDayCos'] = np.cos(2 * np.pi * df['HourOfDay'] / 24)
            df['DayOfYearSin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
            df['DayOfYearCos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
            df['WeekOfYearSin'] = np.sin(2 * np.pi * df['WeekOfYear'] / 52)
            df['WeekOfYearCos'] = np.cos(2 * np.pi * df['WeekOfYear'] / 52)
            df['DayOfWeekSin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
            df['DayOfWeekCos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    def transform(self):

        self.get_folds()
        self.clean_features()
        self.create_datetime_features()
        self.create_aggregation_features()
        self.create_lag_lead_features()
        self.create_rolling_features()
        self.create_target_features()
        self.create_sun_features()

        if self.normalize:
            self.normalize_features()

        return self.df_train, self.df_test
