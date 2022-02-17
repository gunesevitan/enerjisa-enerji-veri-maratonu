import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_feature_importance(df_feature_importance, title, path=None):

    """
    Visualize feature importance of the models

    Parameters
    ----------
    df_feature_importance [pandas.DataFrame of shape (n_features)]: DataFrame of features as index and importance as values
    title (str): Title of the plot
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    df_feature_importance.sort_values(by='mean_importance', inplace=True, ascending=True)

    fig, ax = plt.subplots(figsize=(24, len(df_feature_importance)))
    ax.barh(
        y=df_feature_importance.index,
        width=df_feature_importance['mean_importance'],
        xerr=df_feature_importance['std_importance'],
        align='center',
        ecolor='black',
        capsize=10
    )

    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_learning_curve(training_losses, validation_losses, title, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    training_losses [array-like of shape (n_epochs)]: Array of training losses computed after every epoch
    validation_losses [array-like of shape (n_epochs)]: Array of validation losses computed after every epoch
    title (str): Title of the plot
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    fig, ax = plt.subplots(figsize=(32, 8), dpi=100)
    sns.lineplot(
        x=np.arange(1, len(training_losses) + 1),
        y=training_losses,
        ax=ax,
        label='train_loss'
    )
    if validation_losses is not None:
        sns.lineplot(
            x=np.arange(1, len(validation_losses) + 1),
            y=validation_losses,
            ax=ax,
            label='val_loss'
        )
    ax.set_xlabel('Epochs/Steps', size=15, labelpad=12.5)
    ax.set_ylabel('Loss', size=15, labelpad=12.5)
    ax.tick_params(axis='x', labelsize=12.5, pad=10)
    ax.tick_params(axis='y', labelsize=12.5, pad=10)
    ax.legend(prop={'size': 18})
    ax.set_title(title, size=20, pad=15)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close(fig)


def visualize_predictions(df_predictions, path=None):

    """
    Visualize learning curves of the models

    Parameters
    ----------
    df_predictions [pandas.DataFrame of shape (n_samples, 5)]: DataFrame of datetime, ground-truth and predictions
    path (str or None): Path of the output file (if path is None, plot is displayed with selected backend)
    """

    if 'Generation' in df_predictions.columns:
        # Visualize predictions along with ground-truth for every fold separately if it is training set
        for fold in range(1, 4):
            fold_idx = df_predictions[f'fold{fold}_predictions'].isnull()
            ground_truth = df_predictions.loc[~fold_idx, ['DateTime', 'Generation']].set_index('DateTime')
            predictions = df_predictions.loc[~fold_idx, ['DateTime', f'fold{fold}_predictions']].set_index('DateTime')
            rmse = mean_squared_error(ground_truth, predictions, squared=False)

            fig, ax = plt.subplots(figsize=(32, 8), dpi=100)
            ax.plot(ground_truth, label='Ground-truth')
            ax.plot(predictions, label='Predictions')
            ax.legend(prop={'size': 15})
            ax.tick_params(axis='x', labelsize=12.5)
            ax.tick_params(axis='y', labelsize=12.5)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set_title(f'Training Set Fold {fold} Ground-truth and Predictions - RMSE: {rmse:.6f}', fontsize=15, pad=12)

            if path is None:
                plt.show()
            else:
                plt.savefig(f'{str(path).split(".")[0]}_fold{fold}.png')
                plt.close(fig)

    else:
        # Visualize predictions of every folds together if it is test set
        fig, ax = plt.subplots(figsize=(32, 8), dpi=100)

        for fold in range(1, 5):
            fold_idx = df_predictions[f'fold{fold}_predictions'].isnull()
            predictions = df_predictions.loc[~fold_idx, ['DateTime', f'fold{fold}_predictions']].set_index('DateTime')
            ax.plot(predictions, label=f'Fold {fold}', alpha=0.1 * fold)

        ax.legend(prop={'size': 15})
        ax.tick_params(axis='x', labelsize=12.5)
        ax.tick_params(axis='y', labelsize=12.5)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(f'Test Set Predictions', fontsize=15, pad=12)

        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close(fig)
