# isort: skip_file
"""Entry points for CLI to orchestrate the pipeline."""

import logging

from omegaconf import DictConfig

import hydra

from lstm_autoencoder.data.simulation import simulate_ecg_data

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="main.yaml")
def main(config: DictConfig) -> None:
    """This is a test entry point for demonstration purpose only.

    Args:
        config: project configuration
    """
    logger.info("Simulating ECG data")
    df = simulate_ecg_data(n_beats=300, fs=100)
    print(df.head())
    # TODO: adding preporcessing, modeling, prediction, and evaluation steps
    logger.debug("This is just visible in the log file.")

    logger.info(f"Testkey is: {config['test']['test_key']}")


if __name__ == "__main__":
    # Entry points for debugger
    main()
