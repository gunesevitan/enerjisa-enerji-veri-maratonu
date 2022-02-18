from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim

import settings
import visualization
import training_utils
from torch_datasets import TabularDataset
import torch_modules
import postprocessing


class TabularNeuralNetworkTrainer:

    def __init__(self, features, target, model_parameters, training_parameters):

        self.features = features
        self.target = target
        self.model_parameters = model_parameters
        self.training_parameters = training_parameters

    def train_fn(self, train_loader, model, criterion, optimizer, device, scheduler=None):

        """
        Train given model on given data loader

        Parameters
        ----------
        train_loader (torch.utils.data.DataLoader): Training set data loader
        model (torch.nn.Module): Model to train
        criterion (torch.nn.modules.loss): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Location of the model and inputs
        scheduler (torch.optim.LRScheduler or None): Learning rate scheduler

        Returns
        -------
        train_loss (float): Average training loss after model is fully trained on training set data loader
        """

        print('\n')
        model.train()
        progress_bar = tqdm(train_loader)
        losses = []

        for features, labels in progress_bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            losses.append(loss.item())
            average_loss = np.mean(losses)
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            progress_bar.set_description(f'train_loss: {average_loss:.6f} - lr: {lr:.8f}')

        train_loss = np.mean(losses)
        return train_loss

    def val_fn(self, val_loader, model, criterion, device):

        """
        Validate given model on given data loader

        Parameters
        ----------
        val_loader (torch.utils.data.DataLoader): Validation set data loader
        model (torch.nn.Module): Model to validate
        criterion (torch.nn.modules.loss): Loss function
        device (torch.device): Location of the model and inputs

        Returns
        -------
        val_loss (float): Average validation loss after model is fully validated on validation set data loader
        val_score (float): Root mean squared error calculated between validation set labels and predictions
        """

        model.eval()
        progress_bar = tqdm(val_loader)
        losses = []
        targets = []
        predictions = []

        with torch.no_grad():
            for features, labels in progress_bar:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
                average_loss = np.mean(losses)
                progress_bar.set_description(f'val_loss: {average_loss:.6f}')
                targets += labels.detach().cpu().numpy().tolist()
                predictions += outputs.detach().cpu().numpy().tolist()

        val_loss = np.mean(losses)
        val_target = np.array(targets)
        val_predictions = (np.array(predictions))
        val_score = mean_squared_error(val_target, val_predictions, squared=False)
        return val_loss, val_score, val_predictions

    def train_and_validate(self, df_train, df_test):

        """
        Train and validate on given dataframe with specified configuration

        Parameters
        ----------
        df_train [pandas.DataFrame of shape (25560, n_columns)]: Training dataframe of features, target and folds
        df_test [pandas.DataFrame of shape (744, n_columns)]: Test dataframe of features
        """

        print(f'\n{"-" * 30}\nRunning Tabular Neural Network Model for Training\n{"-" * 30}\n')
        scores = {}
        test_dataset = TabularDataset(df_test.loc[:, self.features].values)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_parameters['data_loader']['test_batch_size'],
            sampler=SequentialSampler(test_dataset),
            pin_memory=True,
            drop_last=False,
            num_workers=self.training_parameters['data_loader']['num_workers']
        )

        for fold in range(1, 5):

            trn_idx, val_idx = df_train.loc[df_train[f'fold{fold}'] == 'train'].index, df_train.loc[df_train[f'fold{fold}'] == 'val'].index
            print(f'\nFold {fold} - Training: {df_train.loc[trn_idx, "DateTime"].min()} - {df_train.loc[trn_idx, "DateTime"].max()} Validation: {df_train.loc[val_idx, "DateTime"].min()} - {df_train.loc[val_idx, "DateTime"].max()}')

            train_dataset = TabularDataset(df_train.loc[trn_idx, self.features].values, df_train.loc[trn_idx, self.target].values)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_parameters['data_loader']['training_batch_size'],
                sampler=RandomSampler(train_dataset),
                pin_memory=True,
                drop_last=False,
                num_workers=self.training_parameters['data_loader']['num_workers']
            )
            # Create validation dataset if current fold has validation index
            if len(val_idx) > 0:
                val_dataset = TabularDataset(df_train.loc[val_idx, self.features].values, df_train.loc[val_idx, self.target].values)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.training_parameters['data_loader']['test_batch_size'],
                    sampler=SequentialSampler(val_dataset),
                    pin_memory=True,
                    drop_last=False,
                    num_workers=self.training_parameters['data_loader']['num_workers']
                )

            # Set model, loss function, device and seed for reproducible results
            training_utils.set_seed(self.training_parameters['random_state'], deterministic_cudnn=self.training_parameters['deterministic_cudnn'])
            device = torch.device(self.training_parameters['device'])
            criterion = getattr(torch_modules, self.training_parameters['loss_function'])(**self.training_parameters['loss_args'])
            model = getattr(torch_modules, self.model_parameters['model_class'])(**self.model_parameters['model_args'])
            if self.model_parameters['model_checkpoint_path'] is not None:
                model_checkpoint_path = self.model_parameters['model_checkpoint_path']
                model.load_state_dict(torch.load(model_checkpoint_path))
            model.to(device)

            # Set optimizer and learning rate scheduler
            optimizer = getattr(optim, self.training_parameters['optimizer'])(model.parameters(), **self.training_parameters['optimizer_args'])
            scheduler = getattr(optim.lr_scheduler, self.training_parameters['lr_scheduler'])(optimizer, **self.training_parameters['lr_scheduler_args']) if self.training_parameters['lr_scheduler'] is not None else None

            early_stopping = False
            summary = {
                'train_loss': [],
                'val_loss': []
            }

            for epoch in range(1, self.training_parameters['epochs'] + 1):

                if early_stopping:
                    break

                if self.training_parameters['lr_scheduler'] == 'ReduceLROnPlateau':
                    train_loss = self.train_fn(train_loader, model, criterion, optimizer, device, scheduler=None)
                    if len(val_idx) > 0:
                        # Step on validation loss if there is validation set and learning rate scheduler is ReduceLROnPlateau
                        val_loss, val_score, val_predictions = self.val_fn(val_loader, model, criterion, device)
                        val_predictions = postprocessing.clip_negative_values(predictions=val_predictions)
                        val_predictions = postprocessing.clip_night_values(predictions=val_predictions, night_mask=(df_train.loc[val_idx, 'IsBetweenDawnAndDusk'] == 0))
                        scheduler.step(val_loss)
                else:
                    train_loss = self.train_fn(train_loader, model, criterion, optimizer, device, scheduler)
                    if len(val_idx) > 0:
                        # Learning rate scheduler will work in validation function if it is not ReduceLROnPlateau
                        val_loss, val_score, val_predictions = self.val_fn(val_loader, model, criterion, device)
                        val_predictions = postprocessing.clip_negative_values(predictions=val_predictions)
                        val_predictions = postprocessing.clip_night_values(predictions=val_predictions, night_mask=(df_train.loc[val_idx, 'IsBetweenDawnAndDusk'] == 0))

                if len(val_idx) > 0:
                    print(f'Epoch {epoch} - Training Loss: {train_loss:.6f} - Validation Loss: {val_loss:.6f} RMSE: {val_score:.6f}')
                else:
                    print(f'Epoch {epoch} - Training Loss: {train_loss:.6f}')

                if len(val_idx) > 0:
                    best_val_loss = np.min(summary['val_loss']) if len(summary['val_loss']) > 0 else np.inf
                    if val_loss < best_val_loss:
                        # Save model, validation and test set predictions if validation loss improves
                        model_path = settings.MODELS / self.model_parameters['model_filename'] / f'model_fold{fold}.pt'
                        torch.save(model.state_dict(), model_path)
                        print(f'Saving model to {model_path} (validation loss decreased from {best_val_loss:.6f} to {val_loss:.6f})')

                        scores[fold] = val_score
                        df_train.loc[val_idx, f'fold{fold}_predictions'] = val_predictions
                        test_predictions = []
                        with torch.no_grad():
                            for features in tqdm(test_loader):
                                features = features.to(device)
                                outputs = model(features)
                                test_predictions += outputs.detach().cpu().numpy().tolist()

                else:
                    # Save model and test predictions every epoch if there is no validation set
                    model_path = settings.MODELS / self.model_parameters['model_filename'] / f'model_fold{fold}.pt'
                    torch.save(model.state_dict(), model_path)
                    print(f'Saving model to {model_path}')

                    test_predictions = []
                    with torch.no_grad():
                        for features in tqdm(test_loader):
                            features = features.to(device)
                            outputs = model(features)
                            test_predictions += outputs.detach().cpu().numpy().tolist()

                test_predictions = postprocessing.clip_negative_values(predictions=np.array(test_predictions))
                test_predictions = postprocessing.clip_night_values(predictions=test_predictions, night_mask=(df_test.loc[:, 'IsBetweenDawnAndDusk'] == 0))
                df_test.loc[:, f'fold{fold}_predictions'] = test_predictions

                summary['train_loss'].append(train_loss)
                if len(val_idx) > 0:
                    summary['val_loss'].append(val_loss)
                    best_iteration = np.argmin(summary['val_loss']) + 1
                    if len(summary['val_loss']) - best_iteration >= self.training_parameters['early_stopping_patience']:
                        print(f'Early stopping (validation loss didn\'t increase for {self.training_parameters["early_stopping_patience"]} epochs/steps)')
                        print(f'Best validation loss is {np.min(summary["val_loss"]):.6f}')
                        early_stopping = True

            visualization.visualize_learning_curve(
                training_losses=summary['train_loss'],
                validation_losses=summary['val_loss'],
                title=f'{self.model_parameters["model_filename"]} Fold {fold} - Learning Curve',
                path=settings.MODELS / self.model_parameters['model_filename'] / f'learning_curve_fold{fold}.png'
            )

        # Display scores of validation splits
        print('\n')
        for fold, score in scores.items():
            print(f'Fold {fold} - Validation RMSE: {score:.6f}')
        print(f'{"-" * 30}\nTabular Neural Network Mean Validation Score: {np.mean(list(scores.values())):6f} (±{np.std(list(scores.values())):2f})\n{"-" * 30}\n')
        # Save validation and test predictions along with DateTime
        df_train[['DateTime', 'Generation'] + [f'fold{fold}_predictions' for fold in range(1, 4)]].to_csv(settings.MODELS / self.model_parameters['model_filename'] / 'train_predictions.csv', index=False)
        visualization.visualize_predictions(
            df_predictions=df_train[['DateTime', 'Generation'] + [f'fold{fold}_predictions' for fold in range(1, 4)]],
            path=settings.MODELS / self.model_parameters['model_filename'] / 'train_predictions.png'
        )
        df_test[['DateTime'] + [f'fold{fold}_predictions' for fold in range(1, 5)]].to_csv(settings.MODELS / self.model_parameters['model_filename'] / 'test_predictions.csv', index=False)
        visualization.visualize_predictions(
            df_predictions=df_test[['DateTime'] + [f'fold{fold}_predictions' for fold in range(1, 5)]],
            path=settings.MODELS / self.model_parameters['model_filename'] / 'test_predictions.png'
        )
