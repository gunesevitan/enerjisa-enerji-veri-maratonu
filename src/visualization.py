import matplotlib.pyplot as plt


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
