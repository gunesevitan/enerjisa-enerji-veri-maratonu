import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

import settings
import visualization
import postprocessing


class LightGBMTrainer:

    def __init__(self, features, target, model_parameters, fit_parameters, categorical_features):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters
        self.categorical_features = categorical_features

    def train_and_validate(self, df_train, df_test):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df_train [pandas.DataFrame of shape (25560, n_columns)]: Training dataframe of features, target and folds
        df_test [pandas.DataFrame of shape (744, n_columns)]: Test dataframe of features
        """

        print(f'{"-" * 30}\nRunning LightGBM Model for Training\n{"-" * 30}\n')
        scores = {}
        df_feature_importance = pd.DataFrame(
            data=np.zeros((len(self.features), 4)),
            index=self.features,
            columns=[f'fold_{fold}_importance' for fold in range(1, 5)]
        )

        for fold in range(1, 5):

            trn_idx, val_idx = df_train.loc[df_train[f'fold{fold}'] == 'train'].index, df_train.loc[df_train[f'fold{fold}'] == 'val'].index
            print(f'Fold {fold} - Training: {df_train.loc[trn_idx, "DateTime"].min()} - {df_train.loc[trn_idx, "DateTime"].max()} Validation: {df_train.loc[val_idx, "DateTime"].min()} - {df_train.loc[val_idx, "DateTime"].max()}')

            trn_dataset = lgb.Dataset(df_train.loc[trn_idx, self.features], label=df_train.loc[trn_idx, self.target], categorical_feature=self.categorical_features)
            if len(val_idx) > 0:
                val_dataset = lgb.Dataset(df_train.loc[val_idx, self.features], label=df_train.loc[val_idx, self.target], categorical_feature=self.categorical_features)
            else:
                val_dataset = None

            model = lgb.train(
                params=self.model_parameters,
                train_set=trn_dataset,
                valid_sets=[trn_dataset, val_dataset] if val_dataset is not None else [trn_dataset],
                num_boost_round=self.fit_parameters['boosting_rounds'],
                callbacks=[
                    lgb.early_stopping(self.fit_parameters['early_stopping_rounds']),
                    lgb.log_evaluation(self.fit_parameters['verbose_eval'])
                ]
            )
            model.save_model(
                settings.MODELS / 'lightgbm' / f'model_fold{fold}.txt',
                num_iteration=None,
                start_iteration=0,
                importance_type='gain'
            )
            df_feature_importance[f'fold_{fold}_importance'] = model.feature_importance(importance_type='gain')

            if len(val_idx) > 0:
                val_predictions = postprocessing.clip_negative_values(predictions=model.predict(df_train.loc[val_idx, self.features]))
                val_predictions = postprocessing.clip_night_values(predictions=val_predictions, night_mask=(df_train.loc[val_idx, 'IsBetweenDawnAndDusk'] == 0))
                df_train.loc[val_idx, f'lgb_fold{fold}_predictions'] = val_predictions
                val_score = mean_squared_error(df_train.loc[val_idx, self.target], np.clip(df_train.loc[val_idx, f'lgb_fold{fold}_predictions'], a_min=0, a_max=None), squared=False)
                scores[fold] = val_score
                print(f'\nLightGBM Validation RMSE: {val_score:.6f}\n')

            test_predictions = postprocessing.clip_negative_values(model.predict(df_test[self.features]))
            test_predictions = postprocessing.clip_night_values(predictions=test_predictions, night_mask=(df_test['IsBetweenDawnAndDusk'] == 0))
            df_test[f'lgb_fold{fold}_predictions'] = test_predictions

        print('\n')
        for fold, score in scores.items():
            print(f'Fold {fold} - Validation RMSE: {score:.6f}')
        print(f'{"-" * 30}\nLightGBM Mean Validation Score: {np.mean(list(scores.values())):6f} (±{np.std(list(scores.values())):2f})\n{"-" * 30}\n')

        df_test[['DateTime'] + [f'lgb_fold{fold}_predictions' for fold in range(1, 5)]].to_csv(settings.MODELS / 'lightgbm' / 'test_predictions.csv', index=False)
        df_feature_importance['mean_importance'] = df_feature_importance[[f'fold_{fold}_importance' for fold in range(1, 5)]].mean(axis=1)
        df_feature_importance['std_importance'] = df_feature_importance[[f'fold_{fold}_importance' for fold in range(1, 5)]].std(axis=1)
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='LightGBM - Feature Importance (Gain)',
            path=settings.MODELS / 'lightgbm' / 'feature_importance.png'
        )
