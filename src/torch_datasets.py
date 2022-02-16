from torch.utils.data import Dataset
import torch


class TabularDataset(Dataset):

    def __init__(self, features, labels=None):

        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx (int): Index of the sample (0 <= idx < len(self.features))

        Returns
        -------
        features [torch.FloatTensor of shape (1, n_features)]: Features
        label [torch.FloatTensor of shape (1)]: Label
        """

        features = self.features[idx]
        features = torch.tensor(features, dtype=torch.float)

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.float)
            return features, label
        else:
            return features
