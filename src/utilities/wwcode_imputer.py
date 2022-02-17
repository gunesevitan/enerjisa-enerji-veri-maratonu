import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

sys.path.append('..')
import settings


if __name__ == '__main__':

    df_train = pd.read_pickle(settings.DATA / 'train.pkl')
    df_test = pd.read_pickle(settings.DATA / 'test.pkl')

    # Create basic datetime features
    for df in [df_train, df_test]:
        df['Date'] = pd.to_datetime(df['DateTime'].dt.date)
        df['Time'] = df['DateTime'].dt.time
        df['Year'] = df['DateTime'].dt.year
        df['Month'] = df['DateTime'].dt.month
        df['DayOfMonth'] = df['DateTime'].dt.day
        df['HourOfDay'] = df['DateTime'].dt.hour
        df['WeekOfYear'] = df['DateTime'].dt.isocalendar().week.astype(np.int64)
        df['DayOfWeek'] = df['DateTime'].dt.dayofweek
        df['DayOfYear'] = df['DateTime'].dt.dayofyear

    # Remove rows with missing WWCode and group their values according to this document
    # https://allaboutweather.tripod.com/presentweather.htm
    df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    df_all = df_all.loc[~df_all['WWCode'].isnull(), :]
    df_all.loc[(df_all['WWCode'] >= 0) & (df_all['WWCode'] < 20), 'WWCode_Group'] = '0-19'
    df_all.loc[(df_all['WWCode'] >= 20) & (df_all['WWCode'] < 30), 'WWCode_Group'] = '20-29'
    df_all.loc[(df_all['WWCode'] >= 30) & (df_all['WWCode'] < 40), 'WWCode_Group'] = '30-39'
    df_all.loc[(df_all['WWCode'] >= 40) & (df_all['WWCode'] < 50), 'WWCode_Group'] = '40-49'
    df_all.loc[(df_all['WWCode'] >= 50) & (df_all['WWCode'] < 60), 'WWCode_Group'] = '50-59'
    df_all.loc[(df_all['WWCode'] >= 60) & (df_all['WWCode'] < 70), 'WWCode_Group'] = '60-69'
    df_all.loc[(df_all['WWCode'] >= 70) & (df_all['WWCode'] < 80), 'WWCode_Group'] = '70-79'
    df_all.loc[(df_all['WWCode'] >= 80) & (df_all['WWCode'] < 91), 'WWCode_Group'] = '80-90'
    df_all.loc[(df_all['WWCode'] >= 91) & (df_all['WWCode'] < 95), 'WWCode_Group'] = '91-95'
    df_all.loc[(df_all['WWCode'] >= 95) & (df_all['WWCode'] < 100), 'WWCode_Group'] = '95-100'
    label_encoder = LabelEncoder()
    df_all['WWCode_Group'] = label_encoder.fit_transform(df_all['WWCode_Group'])

    # Using a simple train/test split to evaluate LightGBM imputer performance
    df_train_wwcode, df_test_wwcode = train_test_split(
        df_all,
        test_size=0.2,
        train_size=0.8,
        random_state=42,
        shuffle=True,
        stratify=df_all['WWCode_Group']
    )
    wwcode_predictors = [
        'AirTemperature', 'ComfortTemperature', 'RelativeHumidity', 'EffectiveCloudCover',
        'Month', 'DayOfMonth', 'HourOfDay', 'DayOfYear'
    ]
    categorical_features = []
    trn_dataset = lgb.Dataset(df_train_wwcode.loc[:, wwcode_predictors], label=df_train_wwcode.loc[:, 'WWCode_Group'], categorical_feature=categorical_features)
    test_dataset = lgb.Dataset(df_test_wwcode.loc[:, wwcode_predictors], label=df_test_wwcode.loc[:, 'WWCode_Group'], categorical_feature=categorical_features)

    # A multi-class classification LightGBM model is trained for WWCode imputation
    model = lgb.train(
        params={
            'num_leaves': 32,
            'learning_rate': 0.01,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'feature_fraction': 0.9,
            'feature_fraction_bynode': 0.9,
            'min_data_in_leaf': 100,
            'min_gain_to_split': 0,
            'lambda_l1': 0,
            'lambda_l2': 0,
            'max_bin': 255,
            'max_depth': -1,
            'objective': 'multiclass',
            'num_class': 10,
            'seed': 42,
            'feature_fraction_seed': 42,
            'bagging_seed': 42,
            'drop_seed': 42,
            'data_random_seed': 42,
            'boosting_type': 'gbdt',
            'verbose': 1,
            'metric': 'multi_logloss',
            'n_jobs': -1
        },
        train_set=trn_dataset,
        valid_sets=[trn_dataset, test_dataset],
        num_boost_round=5000,
        callbacks=[
            lgb.early_stopping(250),
            lgb.log_evaluation(100)
        ]
    )

    test_wwcode_predictions = np.argmax(model.predict(df_test_wwcode.loc[:, wwcode_predictors]), axis=1)
    test_score = accuracy_score(df_test_wwcode['WWCode_Group'], test_wwcode_predictions)
    print(f'\nLightGBM WWCode Imputer - Test Accuracy: {test_score:.6f}')

    # Fill missing values with LightGBM imputer and replace grouped values using label encoder's lookup table
    missing_wwcode_predictions = pd.Series(np.argmax(model.predict(df_train.loc[df_train['WWCode'].isnull(), wwcode_predictors]), axis=1))
    df_train['WWCode_Filled'] = df_train['WWCode']
    df_train.loc[df_train['WWCode'].isnull(), 'WWCode_Filled'] = missing_wwcode_predictions.apply(lambda x: int(label_encoder.classes_[x].split('-')[0])).values
    df_train['WWCode_Filled'] = df_train['WWCode_Filled'].astype(int)
    df_train[['DateTime', 'WWCode_Filled']].to_csv(settings.DATA / 'wwcode_filled.csv', index=False)
