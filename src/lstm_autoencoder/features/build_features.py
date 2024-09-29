import logging


def build_features(filepath: str) -> None:
    """Building features for input files.

    Runs feature building scripts to turn clean data from (../data/02_prepared).

    Args:
        filepath: dir of input files
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
