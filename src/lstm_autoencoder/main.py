# isort: skip_file
"""Entry points for CLI to orchestrate the pipeline."""

import logging

from omegaconf import DictConfig

import hydra

from lstm_autoencoder.data.simulation import simulate_ecg_data
from lstm_autoencoder.data.preprocessing import scale_data, train_test_val_split
from lstm_autoencoder.data.windowed_dataset import get_windowed_datasets
from lstm_autoencoder.models.autoencoder import train_lstm_autoencoder

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="main.yaml")
def main(config: DictConfig) -> None:
    """This is a test entry point for demonstration purpose only.

    Args:
        config: project configuration
    """
    logger.info("Simulating ECG data")
    df = simulate_ecg_data(n_beats=100, fs=100)
    # taking only ecg_amplitude column for training
    df = df[["ecg_amplitude"]]
    logger.info("Splitting data into train, val, and test sets")
    train, val, test = train_test_val_split(df)
    logger.info("Scaling data")
    scaler_filename = (
        config.data.scaler_name + ".pkl"
        if ".pkl" not in config.data.scaler_name
        else config.data.scaler_name
    )
    train, val, test = scale_data(train, test, val, scaler_path=scaler_filename)
    logger.info("Creating windowed datasets")
    logger.debug("Window prep config: %s", config.data.window_prep)
    tf_train, tf_val, tf_test = get_windowed_datasets(train, val, test, config.data.window_prep)

    logger.info("Training LSTM autoencoder")
    _ = train_lstm_autoencoder(
        tf_train.data_windowed,
        tf_val.data_windowed,
        strategy="auto",
        window_size=config.data.window_prep.window_size,
        train_params=config.model.train_params,
        save_path="",
    )

    logger.info("Model training complete")


if __name__ == "__main__":
    # Entry points for debugger
    main()
