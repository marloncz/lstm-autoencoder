import logging


def make_dataset(filepath: str) -> None:
    """Runs data loading scripts that saves new raw data in (../data/01_raw).

    Args:
            filepath: directory of output files
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
