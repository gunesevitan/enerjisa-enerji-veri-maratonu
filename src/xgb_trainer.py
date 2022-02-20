import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb

import settings
import visualization
import postprocessing


class XGBoostTrainer:

    def __init__(self, features, target, model_parameters, fit_parameters):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.fit_parameters = fit_parameters

    def train_and_validate(self, df_train, df_test):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df_train [pandas.DataFrame of shape (25560, n_columns)]: Training dataframe of features, target and folds
        df_test [pandas.DataFrame of shape (744, n_columns)]: Test dataframe of features
        """

        print(f'{"-" * 30}\nRunning XGBoost Model for Training\n{"-" * 30}\n')
        scores = {}
        df_feature_importance = pd.DataFrame(
            data=np.zeros((len(self.features), 4)),
            index=self.features,
            columns=[f'fold_{fold}_importance' for fold in range(1, 5)]
        )

        for fold in range(1, 5):

            trn_idx, val_idx = df_train.loc[df_train[f'fold{fold}'] == 'train'].index, df_train.loc[df_train[f'fold{fold}'] == 'val'].index
            print(f'Fold {fold} - Training: {df_train.loc[trn_idx, "DateTime"].min()} - {df_train.loc[trn_idx, "DateTime"].max()} Validation: {df_train.loc[val_idx, "DateTime"].min()} - {df_train.loc[val_idx, "DateTime"].max()}')

            trn_dataset = xgb.DMatrix(df_train.loc[trn_idx, self.features], label=df_train.loc[trn_idx, self.target])
            # Create validation dataset if current fold has validation index
            if len(val_idx) > 0:
                val_dataset = xgb.DMatrix(df_train.loc[val_idx, self.features], label=df_train.loc[val_idx, self.target])
            else:
                val_dataset = None

            # Set model parameters, train parameters and callbacks
            model = xgb.train(
                params=self.model_parameters,
                dtrain=trn_dataset,
                evals=[(trn_dataset, 'train'), (val_dataset, 'val')] if val_dataset is not None else [(trn_dataset, 'train')],
                num_boost_round=self.fit_parameters['boosting_rounds'] if fold < 4 else 8000,
                early_stopping_rounds=self.fit_parameters['early_stopping_rounds'],
                verbose_eval=self.fit_parameters['verbose_eval'],
            )
            model.save_model(settings.MODELS / 'xgboost' / f'model_fold{fold}.txt')
            for feature, importance in model.get_score(importance_type='gain').items():
                df_feature_importance.loc[feature, f'fold_{fold}_importance'] = importance

            if len(val_idx) > 0:
                val_predictions = postprocessing.clip_negative_values(predictions=model.predict(xgb.DMatrix(df_train.loc[val_idx, self.features])))
                val_predictions = postprocessing.clip_night_values(predictions=val_predictions, night_mask=(df_train.loc[val_idx, 'IsBetweenDawnAndDusk'] == 0))
                df_train.loc[val_idx, f'fold{fold}_predictions'] = val_predictions
                val_score = mean_squared_error(df_train.loc[val_idx, self.target], df_train.loc[val_idx, f'fold{fold}_predictions'], squared=False)
                scores[fold] = val_score
                print(f'\nXGBoost Validation RMSE: {val_score:.6f}\n')

            test_predictions = postprocessing.clip_negative_values(predictions=model.predict(xgb.DMatrix(df_test[self.features])))
            test_predictions = postprocessing.clip_night_values(predictions=test_predictions, night_mask=(df_test['IsBetweenDawnAndDusk'] == 0))
            df_test[f'fold{fold}_predictions'] = test_predictions

        # Display scores of validation splits
        print('\n')
        for fold, score in scores.items():
            print(f'Fold {fold} - Validation RMSE: {score:.6f}')
        print(f'{"-" * 30}\nXGBoost Mean Validation Score: {np.mean(list(scores.values())):6f} (±{np.std(list(scores.values())):2f})\n{"-" * 30}\n')
        # Save validation and test predictions along with DateTime
        df_train[['DateTime', 'Generation'] + [f'fold{fold}_predictions' for fold in range(1, 4)]].to_csv(settings.MODELS / 'xgboost' / 'train_predictions.csv', index=False)
        visualization.visualize_predictions(
            df_predictions=df_train[['DateTime', 'Generation'] + [f'fold{fold}_predictions' for fold in range(1, 4)]],
            path=settings.MODELS / 'xgboost' / 'train_predictions.png'
        )
        df_test[['DateTime'] + [f'fold{fold}_predictions' for fold in range(1, 5)]].to_csv(settings.MODELS / 'xgboost' / 'test_predictions.csv', index=False)
        visualization.visualize_predictions(
            df_predictions=df_test[['DateTime'] + [f'fold{fold}_predictions' for fold in range(1, 5)]],
            path=settings.MODELS / 'xgboost' / 'test_predictions.png'
        )
        df_feature_importance['mean_importance'] = df_feature_importance[[f'fold_{fold}_importance' for fold in range(1, 5)]].mean(axis=1)
        df_feature_importance['std_importance'] = df_feature_importance[[f'fold_{fold}_importance' for fold in range(1, 5)]].std(axis=1)
        visualization.visualize_feature_importance(
            df_feature_importance=df_feature_importance,
            title='XGBoost - Feature Importance (Gain)',
            path=settings.MODELS / 'xgboost' / 'feature_importance.png'
        )
