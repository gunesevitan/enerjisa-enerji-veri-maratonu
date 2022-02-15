import sys
import numpy as np
import pandas as pd

sys.path.append('..')
import settings


def create_folds(df, verbose=True):

    """
    Create a columns of train and validation splits on given training set

    Parameters
    ----------
    df [pandas.DataFrame of shape (25560, 9)]: Training set
    verbose (bool): Flag for verbosity
    """

    # Fold 1: Training set 2019.01.01-2019.11.30 (334 days) and validation set 2019.12.01-2019.12.31 (31 days)
    df['fold1'] = np.nan
    df.loc[(df['DateTime'] >= '2019-01-01 00:00:00') & (df['DateTime'] < '2019-12-01 00:00:00'), 'fold1'] = 'train'
    df.loc[(df['DateTime'] >= '2019-12-01 00:00:00') & (df['DateTime'] < '2020-01-01 00:00:00'), 'fold1'] = 'val'
    # Fold 2: Training set 2020.01.01-2020.11.30 (335 days) and validation set 2020.12.01-2020.12.31 (31 days)
    df['fold2'] = np.nan
    df.loc[(df['DateTime'] >= '2020-01-01 00:00:00') & (df['DateTime'] < '2020-12-01 00:00:00'), 'fold2'] = 'train'
    df.loc[(df['DateTime'] >= '2020-12-01 00:00:00') & (df['DateTime'] < '2021-01-01 00:00:00'), 'fold2'] = 'val'
    # Fold 3: Training set 2019.01.01-2020.11.30 (700 days) and validation set 2020.12.01-2020.12.31 (31 days)
    df['fold3'] = np.nan
    df.loc[(df['DateTime'] >= '2019-01-01 00:00:00') & (df['DateTime'] < '2020-12-01 00:00:00'), 'fold3'] = 'train'
    df.loc[(df['DateTime'] >= '2020-12-01 00:00:00') & (df['DateTime'] < '2021-01-01 00:00:00'), 'fold3'] = 'val'
    # Fold 4: Training set 2019.01.01-2021.11.30 (1065 days) and no validation set
    df['fold4'] = 'train'

    if verbose:
        print(f'\nTraining set split into 4 training and validation sets')
        for fold in range(1, 5):
            df_fold_train = df.loc[df[f'fold{fold}'] == 'train', :]
            df_fold_val = df.loc[df[f'fold{fold}'] == 'val', :]
            print(f'Fold {fold} Training Set: {df_fold_train.shape} - Generation mean: {df_fold_train["Generation"].mean():.4f} std: {df_fold_train["Generation"].std():.4f} min: {df_fold_train["Generation"].min():.4f} max: {df_fold_train["Generation"].max():.4f}')
            print(f'Fold {fold} Validation Set: {df_fold_val.shape} - Generation mean: {df_fold_val["Generation"].mean():.4f} std: {df_fold_val["Generation"].std():.4f} min: {df_fold_val["Generation"].min():.4f} max: {df_fold_val["Generation"].max():.4f}')


if __name__ == '__main__':

    df_train = pd.read_pickle('../../data/train.pkl')
    create_folds(df_train, verbose=True)
    df_train[['DateTime', 'fold1', 'fold2', 'fold3', 'fold4']].to_pickle(settings.DATA / 'folds.pkl')
