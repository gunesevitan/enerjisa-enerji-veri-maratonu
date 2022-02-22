import pandas as pd

import settings


if __name__ == '__main__':

    df_sample_submission = pd.read_csv(settings.DATA / 'sample_submission.csv', sep=',')
    df_lgb_test_predictions = pd.read_csv(settings.MODELS / 'lightgbm' / 'test_predictions.csv')
    df_sample_submission['Generation'] = df_lgb_test_predictions['fold4_predictions']
    df_sample_submission.to_csv(settings.DATA / 'submission.csv', index=False, sep=',')
