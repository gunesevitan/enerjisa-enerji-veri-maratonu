import argparse
import yaml
import pandas as pd

import settings
import preprocessing
from lgb_trainer import LightGBMTrainer
from tnn_trainer import TabularNeuralNetworkTrainer
from snn_trainer import SequenceNeuralNetworkTrainer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    df_train = pd.read_pickle(settings.DATA / 'train.pkl')
    df_test = pd.read_pickle(settings.DATA / 'test.pkl')

    preprocessor = preprocessing.TabularPreprocessor(
        df_train=df_train,
        df_test=df_test,
        fill_missing_values=config['fill_missing_values'],
        normalize=config['normalize_features']
    )
    df_train, df_test = preprocessor.transform()

    print(f'Training Set Shape: {df_train.shape} - Memory Usage: {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'Test Set Shape: {df_test.shape} - Memory Usage: {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

    if config['model'] == 'lightgbm':

        trainer = LightGBMTrainer(
            features=config['features'],
            target=config['target'],
            model_parameters=config['model_parameters'],
            fit_parameters=config['fit_parameters'],
            categorical_features=config['categorical_features']
        )

    elif config['model'] == 'tabular_neural_network':

        trainer = TabularNeuralNetworkTrainer(
            features=config['features'],
            target=config['target'],
            model_parameters=config['model_parameters'],
            training_parameters=config['training_parameters']
        )

    elif config['model'] == 'convolutional_neural_network':

        trainer = SequenceNeuralNetworkTrainer(
            features=config['features'],
            target=config['target'],
            model_parameters=config['model_parameters'],
            training_parameters=config['training_parameters']
        )

    trainer.train_and_validate(df_train, df_test)
