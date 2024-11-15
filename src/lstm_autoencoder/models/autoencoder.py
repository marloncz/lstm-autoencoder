import logging
from pathlib import Path
from typing import no_type_check

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn import metrics
from torch import nn, utils

from lstm_autoencoder.data.windowed_dataset import WindowedDataset

log = logging.getLogger(__name__)


class WeightedMSELoss(torch.nn.Module):
    """Loss Class for Weighted Mean Squared Error."""

    def __init__(self):
        """Initializes the WeightedMSELoss class."""
        super().__init__()

    def forward(self, y_pred, y_true):
        """Defines the forward pass.

        Args:
            y_pred: predicted values.
            y_true: actual values.

        Returns:
            loss: Weighted Mean Squared Error.
        """
        # higher values are more important
        # ensure values above 1
        weight = torch.abs(y_true) + 1
        # weighted MSE
        loss = torch.mean(weight**4 * (y_pred - y_true) ** 2)
        return loss


class Encoder(pl.LightningModule):
    """Encoder class for LSTM Autoencoder.

    This class implements the encoder part of an LSTM-based Autoencoder. It takes time series data as input,
    processes it through two LSTM layers, and outputs a compressed representation (hidden state).
    """

    def __init__(self, input_length: int = 150, n_features: int = 10, compressed_dim: int = 64):
        """Initializes the encoder class.

        Args:
            input_length: The length of the input sequence, typically representing the window size. Defaults to 150.
            n_features: The number of input features (dimensions) that will be encoded and decoded. Defaults to 10.
            compressed_dim: The number of units in the compressed hidden state (final LSTM output). Defaults to 64.
        """
        super().__init__()
        self.input_length, self.n_features = input_length, n_features
        self.compressed_dim, self.hidden_dimension = compressed_dim, 2 * compressed_dim

        self.enc1 = nn.LSTM(
            input_size=self.n_features,  # Default 10
            hidden_size=self.hidden_dimension,  # Default 128
            batch_first=True,
        )

        self.enc2 = nn.LSTM(
            input_size=self.hidden_dimension,  # Default 128
            hidden_size=self.compressed_dim,  # Default 64
            batch_first=True,
        )

    def forward(self, x):
        """Defines the forward pass of the encoder.

        Args:
            x: A batch of input sequences with shape (batch_size, input_length, n_features).

        Returns:
            A tensor representing the compressed hidden state (latent representation)
            with shape (batch_size, compressed_dim).
        """
        x, (_, _) = self.enc1(x)
        _, (hidden, _) = self.enc2(x)
        x = hidden.transpose(0, 1)
        return x


class Decoder(pl.LightningModule):
    """Decoder class for LSTM Autoencoder.

    This class implements the decoder part of an LSTM-based Autoencoder. It takes the compressed (latent) representation
    and reconstructs the original time series data using two LSTM layers and a dense layer.
    """

    def __init__(self, input_length: int = 150, n_features: int = 10, compressed_dim: int = 64):
        """Initializes the decoder class.

        Args:
            input_length: The length of the input sequence to be reconstructed (window size). Defaults to 150.
            n_features: The number of features (dimensions) of the output time series data. Defaults to 10.
            compressed_dim: The number of units in the compressed hidden state (input to the decoder). Defaults to 64.
        """
        super().__init__()
        self.input_length, self.n_features = input_length, n_features
        self.compressed_dimension, self.hidden_dimension = (
            compressed_dim,
            2 * compressed_dim,
        )

        self.dec1 = nn.LSTM(
            input_size=self.compressed_dimension,  # Default 64
            hidden_size=self.compressed_dimension,  # Default 64
            batch_first=True,
        )

        self.dec2 = nn.LSTM(
            input_size=self.compressed_dimension,  # Default 64
            hidden_size=self.hidden_dimension,  # Default 128
            batch_first=True,
        )

        self.dense_layer = nn.Linear(self.hidden_dimension, self.n_features)

    def forward(self, x):
        """Defines the forward pass of the decoder.

        Args:
            x: A batch of latent (compressed) representations with shape (batch_size, 1, compressed_dim).

        Returns:
            A tensor representing the reconstructed time series data with shape
            (batch_size, input_length, n_features).
        """
        x = x.repeat(1, self.input_length, 1)
        x, (_, _) = self.dec1(x)
        x, (_, _) = self.dec2(x)
        x = self.dense_layer(x)
        return x


class Autoencoder(pl.LightningModule):
    """Autoencoder class that combines an Encoder and a Decoder to perform dimensionality reduction and reconstruction of time series data.

    This model is based on LSTM layers and is implemented as a PyTorch Lightning module,
    which includes the training and validation steps.
    """

    def __init__(
        self,
        input_length: int = 150,
        n_features_in: int = 10,
        n_features_out: int = 10,
        compressed_dim: int = 64,
    ):
        """Initializes the Autoencoder with an encoder, decoder, and loss function.

        Args:
            input_length: The length of the input sequence (window size). Defaults to 150.
            n_features_in: The number of input features for the encoder. Defaults to 10.
            n_features_out: The number of output features for the decoder. Defaults to 10.
            compressed_dim: The number of units in the compressed hidden state (latent space). Defaults to 64.
        """
        super().__init__()

        self.encoder = Encoder(input_length, n_features_in, compressed_dim)
        self.decoder = Decoder(input_length, n_features_out, compressed_dim)
        self.loss = WeightedMSELoss()
        self.feature_names = None

    def forward(self, x):
        """Defines the forward pass of the autoencoder, passing the input through the encoder and decoder to reconstruct the original input.

        Args:
            x: A batch of input sequences with shape (batch_size, input_length, n_features_in).

        Returns:
            A tensor representing the reconstructed time series data with shape
            (batch_size, input_length, n_features_out).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        """Here you compute and return the training loss and some additional metrics for e.g. the progress bar or logger.

        Args:
            batch (~torch.Tensor | (~torch.Tensor, ...) | [~torch.Tensor, ...]):
                The output of your ~torch.utils.data.DataLoader. A tensor, tuple or list.
            batch_idx (int): Integer displaying index of this batch

        Return:
            Any of.
                - ~torch.Tensor: The loss tensor
                - dict: A dictionary. Can include any keys, but must include the key 'loss'
                - None: Training will skip to the next batch. This is only for automatic optimization. This is not supported for multi-GPU, TPU, IPU, or DeepSpeed.
        """
        x = batch[0]
        y = batch[1]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss.item(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Computes the validation loss for a given batch and logs the result.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing input data (x) and target data (y).
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the current validation step.
        """
        x = batch[0]
        y = batch[1]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("val_loss", loss.item(), sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer for the autoencoder.

        Returns:
            torch.optim.Adam: The Adam optimizer with a learning rate of 0.00008.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00008)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }


@no_type_check
def train_lstm_autoencoder(
    dataset_train: torch.utils.data.Dataset,
    dataset_val: torch.utils.data.Dataset,
    window_size: int,
    train_params: dict,
    save_name: str = "trained_autoencoder",
    save_path: str | None = None,
    strategy: str = "ddp_notebook",
    compression_factor: float = 1.25,
) -> Autoencoder:
    """Trains LSTM autoencoder.

    Args:
        dataset_train (torch.utils.data.Dataset): Torch train dataset.
        dataset_val (torch.utils.data.Dataset): Torch val dataset.
        window_size (int): Window size for sequences.
        train_params (dict): Training parameters.
        save_name (str): Filename suffix of saved model.
        strategy (str): Training strategy.

    Returns:
        Autoencoder: Trained model.

    """
    batch_size = train_params["batch_size"]  # batch size for training
    shuffle = train_params["shuffle"]  # shuffle data for training
    min_epochs = train_params["min_epochs"]  # min epochs for training
    max_epochs = train_params["max_epochs"]  # max epochs for training
    train_device = train_params["train_device"]  # torch train device ("cpu", ...)
    train_workers = train_params["train_workers"]  # number of devices for training.
    load_workers = train_params["load_workers"]  # number of cpu cores for loading
    model_name = f"autoencoder_{save_name}"

    train_loader = utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        # num_workers=load_workers,
        # persistent_workers=persistent_workers,
    )

    val_loader = utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=load_workers,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.005, patience=3, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        strategy=strategy,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        log_every_n_steps=1,
        accelerator=train_device,
        devices=train_workers,
        reload_dataloaders_every_n_epochs=max_epochs,
        check_val_every_n_epoch=2,
        callbacks=[early_stop_callback],
        default_root_dir="./logs",
    )

    # Define Model
    x = next(iter(train_loader))
    n_features_in = x[0].shape[2]
    n_features_out = x[1].shape[2]
    compressed_dim = round(window_size * n_features_in / compression_factor)
    log.info("Compressed dimension: %s", compressed_dim)
    model = Autoencoder(
        input_length=window_size,
        n_features_in=n_features_in,
        n_features_out=n_features_out,
        compressed_dim=compressed_dim,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # save model
    local_model_path = f"{save_path}/{model_name}.pth" if save_path else f"{model_name}.pth"
    torch.save(model.state_dict(), local_model_path)

    return model


@no_type_check
def create_prediction(
    model: Autoencoder,
    test_data: WindowedDataset,
    save_name: str,
    folder_path: str | None = None,
    save_fig: bool = False,
    use_averaging: bool = True,
) -> pd.DataFrame:
    """Creates prediction based on LSTM autoencoder.

    Args:
        model (Autoencoder): Trained model.
        test_data (WindowedDataset): Windowed test dataset.
        save_name (str): Save name for evaluation results.

    Returns:
        list: Predictions.
    """
    test_loader = utils.data.DataLoader(test_data.data_windowed, batch_size=1, shuffle=False)
    log.debug("Created DataLoader")
    with torch.no_grad():
        y_pred, y = [], []
        for batch in test_loader:
            x = batch[0]
            y_batch = batch[1]
            y_pred_batch = model(x)
            y_pred.append(y_pred_batch)
            y.append(y_batch)

    if save_fig:
        fig = plot_sample_predictions_range(
            y_pred=y_pred,
            y=y,
            test_data=test_data,
            sample_idx_start=0,
            sample_idx_end=512,
            features=test_data.main_features + test_data.aux_features,
        )

        log.debug("Created Sample Predictions Plot")

        if folder_path:
            Path(folder_path).mkdir(exist_ok=True)
            save_path = f"{folder_path}/{save_name}_pred_range.png"
        else:
            save_path = f"{save_name}_pred_range.png"
        fig.savefig(save_path)

    pred_df = test_data.windows_to_df(windows=y_pred, use_averaging=use_averaging)
    pred_df.columns = pred_df.columns + "_pred"
    test_df = test_data.windows_to_df(windows=y, use_averaging=use_averaging)
    out = test_df.merge(pred_df, left_index=True, right_index=True)

    # TODO: implement function for loading splitter...
    # data_splitter = get_splitter(
    #     path="data_splitter.pkl",
    # )

    # log.info("Inverse transform with splitter...")

    # pred_df = data_splitter.inverse_transform(pred_df)
    # pred_df = apply_scaler(pred_df, inverse=True)
    # pred_df = data_splitter.drop_aux_features(pred_df)

    return out


@no_type_check
def plot_sample_predictions_range(
    y_pred: list,
    y: list,
    test_data: WindowedDataset,
    sample_idx_start: int,
    sample_idx_end: int,
    features: list[str] | None = None,
) -> plt.figure:
    """Plot sample predictions over range.

    Args:
        y_pred (list): Predicted values.
        y (list): True values.
        test_data (WindowedDataset): Windowed test dataset.
        sample_idx_start (int): First sample idx (within batch) to be plotted.
        sample_idx_end (int): Last sample idx (within batch) to be plotted.
        features (Optional[List[str]]): List of features used in respective model.

    Returns:
        plt.figure: Sample prediction figure.

    """
    y_pred = test_data.windows_to_df(windows=y_pred)
    y = test_data.windows_to_df(windows=y)

    r2_scores = abs(metrics.r2_score(y, y_pred, multioutput="raw_values"))
    n_features = y_pred.shape[1]

    fig, axs = plt.subplots(n_features + 1, 1, figsize=(16, (n_features + 1) * 1.25), sharex=True)
    for feature_idx in range(n_features):
        axs[feature_idx].plot(
            y_pred.iloc[sample_idx_start:sample_idx_end, feature_idx], c=(0.8, 0.2, 0.2)
        )
        axs[feature_idx].plot(
            y.iloc[sample_idx_start:sample_idx_end, feature_idx], c=(0.2, 0.2, 0.2)
        )
        x_lim = axs[feature_idx].get_xlim()
        x_range = sum(abs(np.array(x_lim)))
        y_lim = axs[feature_idx].get_ylim()
        y_range = sum(abs(np.array(y_lim)))
        # Features
        y_pos = y_lim[1] - 0.2 * y_range
        axs[feature_idx].text(0, y_pos, features[feature_idx])
        # R2
        x_pos = x_lim[1] - 0.05 * x_range
        axs[feature_idx].text(x_pos, y_pos, f"R2={round(r2_scores[feature_idx], 2)}")
        axs[feature_idx].spines["top"].set_visible(False)
        axs[feature_idx].spines["right"].set_visible(False)

    axs[n_features].axis("off")
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    from lstm_autoencoder.data.preprocessing import scale_data, train_test_val_split
    from lstm_autoencoder.data.simulation import simulate_ecg_data
    from lstm_autoencoder.data.windowed_dataset import get_windowed_datasets

    df = simulate_ecg_data(n_beats=100, fs=100)
    df = df[["ecg_amplitude"]]
    train, val, test = train_test_val_split(df)
    train, val, test = scale_data(train, test, val, scaler_path="data/02_intermediate/scaler.pkl")

    prep_params = {
        "window_size": 100,
        "window_shift": 1,
        "split_model_method": "kendall",
        "split_model_th": 0.9,
        "split_model_th_aux": 0.9,
    }

    tf_train, tf_val, tf_test = get_windowed_datasets(train, val, test, prep_params)

    train_params = {
        "batch_size": 256,  # 256,  # batch size for training
        "shuffle": False,  # shuffle data for training
        "min_epochs": 10,  # min epochs for training, low for testing
        "max_epochs": 10,  # max epochs for training, low for testing
        "train_device": "cpu",  # e.g. "cpu", "mps", "cuda"
        "train_workers": 1,
        "load_workers": 0,
    }

    model = train_lstm_autoencoder(
        tf_train.data_windowed,
        tf_val.data_windowed,
        strategy="auto",
        window_size=prep_params["window_size"],
        train_params=train_params,
        save_path="data/03_models/",
    )
