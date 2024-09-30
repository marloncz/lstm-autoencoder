import logging
import pickle
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from torch import utils

logger = logging.getLogger(__name__)

AUX_COLS = {
    "holiday",
    "day_sin",
    "day_cos",
    "year_sin",
    "year_cos",
}


class WindowedDataset:
    """The WindowedDataset class facilitates splitting tabular data into overlapping windows and merging them back into a table."""

    def __init__(
        self, data_df: pd.DataFrame, window_size: int, window_shift: int, targets: list[str]
    ):
        """WindowedDataset Constructor.

        Args:
            data_df (pd.DataFrame): Tabular input data
            window_size (int): Window size for sequences.
            window_shift (int): Window shift for sequences.

        """
        logger.info("Initialized WindowDataClass with data of the shape %s", data_df.shape)
        self.data_df = data_df
        self.data_features = data_df.columns
        self.window_size = window_size
        self.window_shift = window_shift

        self.data_windowed = utils.data.TensorDataset()
        self.data_timestamps: list[list[pd.Timestamp]] = []

        self.aux_features = list(AUX_COLS & set(data_df.columns))
        self.main_features = list(data_df.columns.difference(self.aux_features))
        self.targets = targets

    def df_to_windows(self):
        """Splits table into overlapping windows."""
        tensors_in, tensors_out, timestamps = WindowedDataset.create_sequences(
            data_df=self.data_df,
            y_features=self.targets,
            window_size=self.window_size,
            window_shift=self.window_shift,
        )

        self.data_windowed = utils.data.TensorDataset(tensors_in, tensors_out)

        self.data_timestamps = timestamps

    def windows_to_df(self, windows: Sequence[utils.data.TensorDataset]) -> pd.DataFrame:
        """Merges overlapping windows into table.

        Args:
            windows (Sequence[utils.data.TensorDataset]): Windows to be merged

        Returns:
            data_df (pd.DataFrame): Merged table
        """
        timestamps = self.data_timestamps
        features = self.targets
        window_shift = self.window_shift
        for window_idx, window in enumerate(windows):
            if isinstance(window, torch.utils.data.TensorDataset):
                # Extract the first tensor from TensorDataset
                data = (
                    window.tensors[0].numpy().squeeze()
                )  # Extract the first tensor and convert to NumPy array
            elif isinstance(window, torch.Tensor):
                # Directly convert Tensor to NumPy array
                data = window.numpy().squeeze()
            # data = window.numpy().squeeze()
            # data = window.tensors[0].numpy().squeeze()
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            if window_idx == 0:
                data_df = pd.DataFrame(data=data, index=timestamps[window_idx], columns=features)
            else:
                data_df2 = pd.DataFrame(
                    data=data[-window_shift::, :],
                    index=timestamps[window_idx][-window_shift::],
                    columns=features,
                )
                data_df = pd.concat((data_df, data_df2), axis=0)

        return data_df

    @staticmethod
    def create_sequences(
        data_df: pd.DataFrame,
        y_features: list[str],
        window_size: int,
        window_shift: int,
    ) -> tuple:
        """Creates sequences from tabular data.

        Args:
            data_df (pd.DataFrame): Data to split into sequences
            y_features (List[str]): Features that must be reconstructed.
            window_size (int): Length of input sequence. Defaults to 24.
            window_shift (int): Shift of rolling window. Defaults to 0.

        Returns:
            tensors_in (torch.FloatTensor): Sequences
            tensors_out (torch.FloatTensor): Sequences
            timestamps (pd.core.indexes.datetimes.DatetimeIndex): Timestamps for Sequences

        """
        n_windows = np.floor((len(data_df) - window_size) / window_shift + 1).astype(int)
        inds_start = np.flip(
            np.arange(
                len(data_df) - window_size,
                (len(data_df) - window_size) - n_windows * window_shift,
                -window_shift,
            )
        ).astype(int)
        inds_stop = inds_start + int(window_size)

        # keep only relevant features for the y part
        data_df_y = data_df[y_features]

        values_in = []
        values_out = []
        timestamps = []
        for i in range(n_windows):
            values_in.append(data_df[inds_start[i] : inds_stop[i]])
            values_out.append(data_df_y[inds_start[i] : inds_stop[i]])
            timestamps.append(data_df.index[inds_start[i] : inds_stop[i]])

        tensors_in = torch.FloatTensor(np.array(values_in))
        tensors_out = torch.FloatTensor(np.array(values_out))

        return tensors_in, tensors_out, timestamps


def create_windowed_datasets(
    data_dfs: Sequence[pd.DataFrame], window_size: int, window_shift: int, targets: list[str]
) -> list[WindowedDataset]:
    """Creates a sequence of WindowedDatasets.

    Args:
        data_dfs (Sequence[pd.Dataframe]): Sequence of dataframes.
        window_size (int): Window size.
        window_shift (int): Shift.

    Returns:
        Sequence[WindowedDatasets]
    """
    datasets = []
    for data_df in data_dfs:
        dataset = WindowedDataset(data_df, window_size, window_shift, targets)
        dataset.df_to_windows()
        datasets.append(dataset)

    return datasets


class DataSplitter:
    """The DataSplitter class facilitates splitting tabular data into multiple tables according to a correlation criterion and merging them back into a single table."""

    def __init__(self, data_df: pd.DataFrame):
        """DataSplitter Constructor.

        Args:
            data_df (pd.DataFrame): Tabular input data

        """
        self.data_df = data_df

    def fit_transform(
        self, method: str = "pearson", th: float = 0.6, th_aux: float = 0.6
    ) -> Sequence[pd.DataFrame]:
        """Computes and applies split.

        Args:
            method (str): Correlation method passed to pd.DataFrame.corr()
                valid method strs: "pearson", "kendall", ...
            th (float): Threshold applied to correlation to determine split of main features
            th_aux (float): Threshold applied to correlation to determine split of aux features

        Returns:
            Sequence[WindowedDatasets]: Split up tabular data
        """
        feature_names, feature_names_aux = self.get_data_split(
            data_df=self.data_df, method=method, th=th, th_aux=th_aux
        )
        data_dfs = self.apply_data_split(self.data_df, feature_names, feature_names_aux)

        self.feature_names = feature_names
        self.feature_names_aux = feature_names_aux

        return data_dfs

    def transform(self, data_df: pd.DataFrame) -> Sequence[pd.DataFrame]:
        """Applies split to table according to object state.

        Args:
            data_df (pd.DataFrame): Tabular input data

        Returns:
            data_dfs (Sequence[pd.DataFrame]): Sequence of tabular data

        """
        data_dfs = self.apply_data_split(
            data_df=data_df,
            feature_names=self.feature_names,
            feature_names_aux=self.feature_names_aux,
        )

        return data_dfs

    def inverse_transform(self, data_dfs: Sequence[pd.DataFrame]) -> pd.DataFrame:
        """Merges split tables according to object state.

        Args:
            data_dfs (Sequence[pd.DataFrame]): Sequence of tabular data

        Returns:
            data_df (pd.DataFrame): Merged tabular data

        """
        data_df = pd.DataFrame(index=data_dfs[0].index)
        for split_idx, split_df in enumerate(data_dfs):
            data_df[self.feature_names[split_idx]] = split_df[self.feature_names[split_idx]]

        self.feature_names_aux_all = self.data_df.columns.difference(data_df.columns)

        data_df[self.feature_names_aux_all] = self.data_df[self.feature_names_aux_all]
        data_df = data_df.reindex(columns=self.data_df.columns)

        return data_df

    def drop_aux_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Removes auxiliary (weather etc.) features from table.

        Args:
            data_df (pd.DataFrame): Tabular data containing aux features

        Returns:
            data_df (pd.DataFrame): Tabular data with aux features removed

        """
        data_df = data_df.drop(labels=self.feature_names_aux_all, axis=1)

        return data_df

    @staticmethod
    def get_correlations(
        df: pd.DataFrame, method: str = "pearson", piece_len: int = 2000
    ) -> pd.DataFrame:
        """Computes piece-wise correlations between columns of dataframe.

        Args:
            df (pd.Dataframe): Dataframe containing data.
            method (str): Correlation method.
            piece_len (int): Piece length to use for piece-wise correlations

        Returns:
            pd.DataFrame: Correlations

        """
        # Piece-wise Correlation
        n_pieces = np.floor(len(df) / piece_len).astype(int)
        corr_ref = df.iloc[0:piece_len].corr(method=method).values
        corr_ref = np.expand_dims(corr_ref, axis=2)
        for piece_idx in np.arange(1, n_pieces + 1):
            row_idx = piece_idx * piece_len
            corr_tar = abs(df.iloc[row_idx : row_idx + piece_len].corr(method=method).values)
            corr_ref = np.concatenate((corr_ref, corr_tar[:, :, np.newaxis]), axis=2)

        corr = pd.DataFrame(
            columns=df.columns,
            index=df.columns,
            data=np.nanmax(abs(corr_ref), axis=2),
        )

        return corr

    @staticmethod
    def get_clusters(
        corr: pd.DataFrame,
        th: float,
    ) -> tuple[int, np.ndarray]:
        """Computes (connected components) clusters given correlation matrix and threshold.

        Args:
            corr (pd.Dataframe): Dataframe containing data.
            th (float): Correlation threshold.

        Returns:
            Tuple:
                n_clusters: Number of resulting clusters
                clusters: Maps columns to cluster ids

        """
        # Compute connected components on thresholded correlation matrix
        corr_th = corr > th
        graph = csr_matrix(corr_th)
        n_clusters, clusters = connected_components(
            csgraph=graph, directed=False, return_labels=True
        )

        return n_clusters, clusters

    @staticmethod
    def get_data_split(
        data_df: pd.DataFrame,
        method: str = "pearson",
        th: float = 0.6,
        th_aux: float = 0.6,
    ) -> tuple:
        """Splits dataframe into multiple dataframes according to their correlation.

        Args:
            data_df (pd.Dataframe): Dataframe containing data.
            method (str): Correlation method.
            th (float): Correlation threshold.
            th_aux (float): Correlation threshold for aux features.

        Returns:
            Tuple:
                data_dfs: Sequence of split up dataframes
                feature_inds: Feature inds corresponding to data split
                feature_names: Feature names corresponding to data split
                feature_names_aux: Auxiliary feature names

        """
        # Split data in main features and auxiliary features
        aux_cols = list(AUX_COLS & set(data_df.columns))
        data_main = data_df[data_df.columns.difference(aux_cols)]
        data_aux = data_df[aux_cols]
        cor_piece = len(data_df)

        # Compute thresholded correlation matrix and compute connected components (clusters)
        feature_names = []
        feature_names_aux = []

        # Get correlations and clusters of main features
        corr = DataSplitter.get_correlations(df=data_main, method=method, piece_len=cor_piece)
        n_clusters, clusters = DataSplitter.get_clusters(corr=corr, th=th)
        logger.info(f"Correlation between Resources:\n{corr}")
        logger.info(f"Defined {n_clusters} based on correlation threshold of {th}")

        for cluster_idx in range(n_clusters):
            feature_idx = np.where(clusters == cluster_idx)[0]
            aux_cols_corr = []

            # Iterate over each feature of main cluster
            for idx in feature_idx:
                # Get main feature of given idx and join with aux features
                data_cluster_main = data_main[data_main.columns[idx]]
                if type(data_cluster_main) is pd.Series:
                    data_cluster_main = data_cluster_main.to_frame()
                data_cluster_all = data_cluster_main.join(data_aux)

                # For each main feature, get correlations to aux features
                corr_aux = DataSplitter.get_correlations(
                    df=data_cluster_all, method=method, piece_len=cor_piece
                )

                logger.debug(f"Correlation between Resource(s) and AUX:\n{corr_aux}")
                logger.debug(f"Threshold for AUX features is {th_aux}")

                # Only keep aux features with corr > th_aux
                aux_cols_corr_idx = np.where(corr_aux.loc[data_main.columns[idx]] > th_aux)[0][1::]
                aux_cols_corr.extend(corr_aux.columns[aux_cols_corr_idx])

            # Only add aux features to which at least one main feature was correlated
            aux_cols_corr = list(set(aux_cols_corr))
            feature_names.append(data_main.columns[feature_idx])
            feature_names_aux.append(aux_cols_corr)

        return feature_names, feature_names_aux

    @staticmethod
    def apply_data_split(
        data_df: pd.DataFrame,
        feature_names: Sequence[pd.Index],
        feature_names_aux: Sequence[pd.Index],
    ) -> Sequence[pd.DataFrame]:
        """Splits dataframe into multiple dataframes according provided feature names.

        Args:
            data_df (pd.Dataframe): Dataframe containing data.
            feature_names (Sequence[pd.Index]): Feature names corresponding to data split
            feature_names_aux (Sequence[pd.Index]): Auxiliary feature names corresponding to data split


        Returns:
            Sequence[pd.DataFrame]: Split up DataFrames

        """
        # Split data in main features and auxiliary features
        aux_cols = list(AUX_COLS & set(data_df.columns))
        data_main = data_df[data_df.columns.difference(aux_cols)]
        data_aux = data_df[aux_cols]

        data_dfs = []
        for feature_name, feature_name_aux in zip(feature_names, feature_names_aux):
            data_dfs.append(data_main[feature_name].join(data_aux[feature_name_aux]))

        return data_dfs


def get_pooled_windowed_datasets(
    data_df_train: pd.DataFrame,
    data_df_val: pd.DataFrame,
    data_df_test: pd.DataFrame,
    prep_params: dict,
    splitter_path: str,
    targets: list[str] = ["ecg_amplitude"],
) -> tuple:
    """Split data and convert to overlapping windows of torch datasets.

    Args:
        data_df_train (pd.DataFrame): Train data.
        data_df_val (pd.DataFrame): Validation data.
        data_df_test (pd.DataFrame): Test data.
        prep_params (dict): Preprocessing parameters.
        splitter_path (str): Path to save splitter object.

    Returns:
        Tuple: Windowed train data, windowed test data
            and DataSplitter object
    """
    window_size = prep_params["window_size"]
    window_shift = prep_params["window_shift"]
    split_model_method = prep_params["split_model_method"]
    split_model_th = prep_params["split_model_th"]
    split_model_th_aux = prep_params["split_model_th_aux"]

    data_splitter = DataSplitter(data_df_train)

    logger.debug("Creating windowed dataset for training")
    data_dfs_train = data_splitter.fit_transform(
        method=split_model_method, th=split_model_th, th_aux=split_model_th_aux
    )

    with open(splitter_path, "wb") as f:
        pickle.dump(data_splitter, f)

    for idx, df in enumerate(data_dfs_train):
        logger.debug(
            "Columns of Dataset %s /  %s: %s", idx, len(data_dfs_train) - 1, df.columns.values
        )

    tf_train = create_windowed_datasets(
        data_dfs=data_dfs_train, window_size=window_size, window_shift=window_shift, targets=targets
    )

    logger.debug("Creating windowed dataset for validation")
    data_dfs_val = data_splitter.transform(data_df_val)
    tf_val = create_windowed_datasets(
        data_dfs=data_dfs_val, window_size=window_size, window_shift=window_shift, targets=targets
    )

    logger.debug("Creating windowed dataset for testing")
    data_dfs_test = data_splitter.transform(data_df_test)
    tf_test = create_windowed_datasets(
        data_dfs=data_dfs_test, window_size=window_size, window_shift=window_shift, targets=targets
    )

    return tf_train, tf_val, tf_test


def get_windowed_datasets(
    data_df_train: pd.DataFrame,
    data_df_val: pd.DataFrame,
    data_df_test: pd.DataFrame | None,
    prep_params: dict,
    targets: list[str] = ["ecg_amplitude"],
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset | None]:
    """Generates windowed datasets for training, validation, and testing without any data splitting.

    Args:
        data_df_train: Dataframe for training.
        data_df_val: Dataframe for validation.
        data_df_test: Dataframe for testing. If not provided, returns None.
        prep_params: Parameters for data preparation.

    Returns:
        Tuple with training, validation, and test datasets. In case of no test data, the third element is None.
    """
    window_size = prep_params["window_size"]
    window_shift = prep_params["window_shift"]

    tf_train = WindowedDataset(data_df_train, window_size, window_shift, targets)
    tf_train.df_to_windows()

    tf_val = WindowedDataset(data_df_val, window_size, window_shift, targets)
    tf_val.df_to_windows()

    if data_df_test is None:
        return tf_train, tf_val, None

    tf_test = WindowedDataset(data_df_test, window_size, window_shift, targets)
    tf_test.df_to_windows()

    return tf_train, tf_val, tf_test
