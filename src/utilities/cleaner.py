import sys
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


if __name__ == '__main__':

    df_generation = pd.read_csv(settings.DATA / 'generation.csv', sep=';')
    df_generation = df_generation.dropna()

    df_temperature = pd.read_csv(settings.DATA / 'temperature.csv', sep=';')
    df_temperature = df_temperature.loc[~df_temperature['DateTime'].isnull(), :]

    df_train = df_generation.merge(df_temperature, how='left', on='DateTime')
    df_sample_submission = pd.read_csv(settings.DATA / 'sample_submission.csv', sep=',')
    df_test = df_sample_submission.merge(df_temperature, how='left', on='DateTime')
    df_test.drop(columns=['Generation'], inplace=True)

    # Assign appropriate data types to columns
    for dataset, df in zip(['train', 'test'], [df_train, df_test]):
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        if dataset == 'train':
            df['Generation'] = df['Generation'].str.replace(',', '.').astype(np.float64)
        df['AirTemperature'] = df['AirTemperature'].str.replace(',', '.').astype(np.float32)
        df['ComfortTemperature'] = df['ComfortTemperature'].str.replace(',', '.').astype(np.float32)
        df['RelativeHumidity'] = df['RelativeHumidity'].str.replace(',', '.').astype(np.float32)
        df['WindSpeed'] = df['WindSpeed'].str.replace(',', '.').astype(np.float32)
        df['WindDirection'] = df['WindDirection'].astype(np.float32)
        df['WWCode'] = df['WWCode'].astype(np.float32)
        df['EffectiveCloudCover'] = df['EffectiveCloudCover'].str.replace(',', '.').astype(np.float32)

    print(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    # Write training and test sets as pickle files
    df_train.to_pickle(settings.DATA / 'train.pkl')
    df_test.to_pickle(settings.DATA / 'test.pkl')
