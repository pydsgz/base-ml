import os
import numpy as np
import pandas as pd
import pickle

import torch
from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from base_ml import utils


class DataProvider(Dataset):
    """Dataset provider for machine learning pipelines.

    Dataset object used by any training workflow.

    Attributes:
        data_x (numpy.ndarray): Input features of size N x F.
        data_y (numpy.ndarray): Input target of size N x T.
        numerical_idx (:obj:`list` of :obj: `int`): List of indeces which are
            numerical features in data_x.
        non_num_idx (:obj:`list` of :obj: `int`): List of indeces which are
            non-numerical features in data_x.
        # TODO make sure variable is not a torch tensor
        adj (numpy.ndarray): Adjacency matrix of size n x n
        # TODO make sure variable is not a torch tensor
        meta_inf (numpy.ndarray): Meta-information of size n x m, this will
            be used to calculate the adjacency matrix.
        args (argparse.ArgumentParser): Argument parser object containing
            configuration for this dataset.
    """
    def __init__(self):
        self.data_x = None
        self.data_y = None
        self.numerical_idx = None
        self.non_num_idx = None
        self.adj = None
        self.meta_inf = None
        self.args = None
        self.binary_idx_list = None
        self.len_numerical_feat = None

    def _load_data(self):
        raise NotImplementedError

    @staticmethod
    def to_one_hot_encoding(target_data):
        """Returns one-hot-encoded array of size N x C

        Args:
            target_data (numpy.ndarray): numpy array containing class labels.
        """
        target_data = target_data.squeeze()
        n_class = len(np.unique(target_data))
        res = np.eye(n_class)[target_data.astype(int)]
        return res

    # TODO improve to property setter
    def set_binary_feature_idx_list(self):
        """List of ints specifying indices for binary features."""
        binary_indices_ = utils.get_binary_indices(self.data_x,
                                                   self.non_num_idx)
        self.binary_idx_list = np.array(binary_indices_)
        self.non_num_idx = np.setdiff1d(self.non_num_idx, self.binary_idx_list)

    # TODO improve to property setter
    def set_len_numerical_feat(self):
        """List of ints specifying indices for numerical features."""
        cur_len_ = len(self.numerical_idx)
        if self.binary_idx_list is not None:
            cur_len_ += len(self.binary_idx_list)
        self.len_numerical_feat = cur_len_


class TransductiveMGMCDataset(Dataset):
    """Dataset provider for matrix completion related training workflow."""
    def __init__(self, feat_data, adj_matrix, target_data, set_idx,
                 val_idx=None, mask_matrix=None, is_training=None,
                 args=None):
        self.adj_mat = adj_matrix
        self.target_data = target_data
        self.idx = set_idx
        self.val_idx = val_idx
        self.mask_matrix = mask_matrix
        self.args = args

        # Use only labels of current set
        target_for_concat = target_data.copy()

        if is_training:
            # Feed training set labels as input
            if args.init_train_labels:
                row_idx = np.setdiff1d(np.arange(target_data.shape[0]), set_idx)
                target_for_concat[row_idx, :] = 0.0

            # Training set labels are initialized to 0.0 # was used for Dizzyreg
            else:
                target_for_concat[:, :] = 0.0
        else:
            # Test set labels set to 0.0
            target_for_concat[:, :] = 0.0

        # Simulate feature level missingness
        data_remain = args.p_remain
        if data_remain != 1.0:
            cur_feat_matrix = feat_data.copy()
            self.original_feat_data = cur_feat_matrix.copy()
            cur_feat_matrix[~mask_matrix] = np.nan
            sim_missing_feat_data, mask_simulated = utils.select_p_data_df(
                cur_feat_matrix, p_select=data_remain,random_seed=0)
            mask_matrix = np.isnan(sim_missing_feat_data)
            sim_missing_feat_data = np.nan_to_num(sim_missing_feat_data, 0.0)
            feat_data = sim_missing_feat_data
            self.missing_idx_sim = mask_simulated

        # Combine feature matrix and label matrix
        self.feat_data = np.concatenate([feat_data, target_for_concat], -1)
        self.feat_data = self.feat_data.astype(np.float32)

        # Class weighting
        class_counts = pd.Series(target_data.argmax(-1)).value_counts()
        self.class_weight = np.array(class_counts.max() / class_counts)

        assert len(feat_data) == len(target_data)

    def __len__(self):
        return len(self.feat_data)

    def __getitem__(self, idx):
        return (self.feat_data[idx], self.adj_mat[idx], idx, self.idx,
                self.val_idx, self.mask_matrix[idx]), self.target_data[idx]


class TransductiveGNNDataset(Dataset):
    """Dataset provider for Transductive GNN classification."""
    def __init__(self, feat_data, adj_matrix, target_data, set_idx,
                 val_idx=None, mask_matrix=None):

        self.feat_data = feat_data
        self.adj_mat = adj_matrix
        self.target_data = target_data
        self.set_idx = set_idx
        self.val_idx = val_idx
        self.mask_matrix = mask_matrix

        # Class weighting
        class_counts = pd.Series(target_data.argmax(-1)).value_counts()
        self.class_weight = np.array(class_counts.max() / class_counts)

        assert len(feat_data) == len(target_data)

    def __len__(self):
        return len(self.feat_data)

    def __getitem__(self, idx):
        ret_val = {'feat_matrix': self.feat_data[idx],
                   'adj_matrix': self.adj_mat[idx],
                   'get_item_index': idx,
                   'set_index': self.set_idx,
                   'val_index': self.val_idx,
                   'mask_matrix': self.mask_matrix[idx]
                   }
        return ret_val, self.target_data[idx]


class InductiveGNNDataset(Dataset):
    """Dataset provider for Inductive learning using Graph Neural Networks"""
    def __init__(self, feat_data, adj_matrix, target_data, mask_matrix):

        self.feat_data = feat_data.astype(np.float32)
        self.adj_mat = adj_matrix.astype(np.float32)
        self.target_data = target_data.argmax(-1)
        self.mask_matrix = mask_matrix

        # Class weighting
        class_counts = pd.Series(target_data.argmax(-1)).value_counts()
        self.class_weight = np.array(class_counts.max() / class_counts)

        assert len(feat_data) == len(target_data)

    def __len__(self):
        return len(self.feat_data)

    def __getitem__(self, idx):
        return_dict = {'feature_matrix': self.feat_data[idx],
                       'adjacency_matrix': self.adj_mat[idx],
                       'index': idx,
                       'mask_matrix': self.mask_matrix[idx],
                       }
        return return_dict, self.target_data[idx]


class MNISTDataset(DataProvider):
    """MNIST classification dataset."""
    def __init__(self, args):
        super(MNISTDataset, self).__init__()
        self.args = args
        self.all_non_numerical_idx = None
        self._load_data()

    def _load_data(self):
        """Load data x and data y."""

        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', cache=True)
        # data_x = np.array(final_data_df)
        feat_data = np.array(mnist.data).astype('float32')
        target_data = mnist.target.astype('int64')
        shuffling_index = np.arange(feat_data.shape[0])
        np.random.shuffle(shuffling_index)
        feat_data = feat_data[shuffling_index]
        target_data = target_data[shuffling_index]

        cur_data_list = []
        cur_target_list = []
        for i in range(10):
            cur_mask = target_data == i
            cur_data_list.append(feat_data[cur_mask][:500])
            cur_target_list.append(target_data[cur_mask][:500])
        feat_data = np.concatenate(cur_data_list)
        target_data = np.concatenate(cur_target_list)

        self.data_x = feat_data
        self.data_y = self.to_one_hot_encoding(target_data)
        self.numerical_idx = np.arange(784)
        self.non_num_idx = None

        # Calculate adjacency matrix
        self.meta_inf = self.data_x.astype('float32')

        if self.args.graph_type:
            self.adj = self.get_adjacency()

    def get_adjacency(self):
        data_path = self.args.output_path
        save_path = os.path.join(data_path, 'adj_matrix_all.pt')
        if self.args.graph_type == 1:
            save_path = save_path.replace('.pt', '_thresholded.pt')
        else:
            raise NotImplementedError

        # Load adjacency matrix if available, otherwise calculate it
        if os.path.exists(save_path):
            with open(save_path, 'rb') as fpath:
                cur_adj = torch.load(fpath)
        else:
            # Calculate adjacency using given similarity metric.
            data_arr = self.meta_inf

            # Thresholded
            threshold_list = [7]
            if self.args.graph_type == 1:
                cur_adj = utils.get_thresholded_eucledian(data_arr,
                                                          threshold_list)
            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj


class DizzyregDataset(DataProvider):
    """Dizzyreg dataset provider."""
    def __init__(self, args, task_num):
        super(DizzyregDataset, self).__init__()
        self.args = args
        self.all_non_numerical_idx = None
        self.task_num = task_num
        self._load_data()
        # We encode all np.nan in binary features as 0.5 so we just treat
        self.set_binary_feature_idx_list()
        self.set_len_numerical_feat()

        # Drop categorical columns which contains 1 unique elem and np.nan

    def _load_data(self):
        """Load data x and data y."""

        path_data_x = '../data/dizzyreg/t%s_df.csv' % self.task_num
        path_data_y = '../data/dizzyreg/label_df_t%s.csv' % self.task_num
        path_meta = '../data/dizzyreg/meta_df_t%s.csv' % self.task_num
        path_numerical_columns = '../data/dizzyreg/num_columns_v2.csv'
        path_nonnumerical_columns = '../data/dizzyreg/non_num_columns_v2.csv'

        read_data_x = pd.read_csv(path_data_x)
        read_data_y = pd.read_csv(path_data_y)
        read_data_meta = pd.read_csv(path_meta)

        # Drop columns if it only contains 1 unique element
        read_data_x = pd.DataFrame(self.drop_one_elem_columns(read_data_x))

        num_col = pd.read_csv(path_numerical_columns)
        num_col = read_data_x.columns.isin(num_col['0'].values).nonzero()[0]
        col_idx = np.arange(read_data_x.shape[-1])
        non_num_col = np.setdiff1d(col_idx, num_col)

        # new_data_x = np.array(read_data_x).astype(np.float32)
        new_data_x = np.array(read_data_x)
        new_data_y = np.array(read_data_y).astype(np.float32)
        new_data_meta = np.array(read_data_meta).astype(np.float32)

        print(new_data_x.shape, new_data_y.shape, new_data_meta.shape)


        # Winsorize dataset
        len_feat = new_data_x.shape[-1]
        idx_list = list(num_col)
        for i in range(len_feat):
            if i in idx_list:
                cur_data = new_data_x[:, i]
                cur_data = np.array(cur_data)
                lower_p = np.percentile(cur_data, 5)
                higher_p = np.percentile(cur_data, 95)
                cur_data[cur_data < lower_p] = lower_p
                cur_data[cur_data > higher_p] = higher_p
                new_data_x[:, i] = cur_data

        # Make sure target data is one-hot encoded
        if new_data_y.shape[-1] == 1:
            num_class = len(np.unique(new_data_y))
            new_data_y = np.eye(num_class)[new_data_y.astype(int).reshape(-1)]
            new_data_y = new_data_y.astype('float32')
        self.orig_column_names = read_data_x.columns
        self.data_x = new_data_x
        self.data_y = new_data_y
        self.numerical_idx = num_col
        self.non_num_idx = non_num_col

        # Calculate adjacency matrix
        self.meta_inf = new_data_meta.astype('float32')
        if self.args.graph_type:
            self.adj = self.get_adjacency()

    def get_adjacency(self):
        data_path = '../data/dizzyreg/'
        save_path = os.path.dirname(data_path)

        save_path = os.path.join(save_path, 'adj_matrix_all.pt')

        if self.args.graph_type == 1:
            save_path = save_path.replace('.pt', '_thresholded.pt')
        elif self.args.graph_type == 2:
            save_path = save_path.replace('.pt', '_knn.pt')
        elif self.args.graph_type == 3:
            save_path = save_path.replace('.pt', '_heom.pt')
        elif self.args.graph_type == 4:
            save_path = save_path.replace('.pt', '_HeomSubset.pt')
        else:
            raise NotImplementedError

        save_path = save_path.replace('.pt', 'task%s.pt' % self.task_num)

        # Load adjacency matrix if available, otherwise calculate it
        if os.path.exists(save_path):
            with open(save_path, 'rb') as fpath:
                cur_adj = torch.load(fpath)
        else:
            # Calculate adjacency using given similarity metric.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cur_meta = torch.from_numpy(self.meta_inf).float().to(device)
            # Thresholded gaussian
            threshold_list = [0, 6, .06, 11]
            # threshold_list = [0, 7.5, .1, 9.22]

            if self.args.graph_type == 1:
                adj_list = []
                for k, v in enumerate(threshold_list):
                    meta_ = cur_meta[:, k:k+1]
                    dist = meta_[:, None, :] - meta_[None, :, :]
                    cur_adj = torch.abs(dist) <= v
                    cur_adj = cur_adj.long()
                    adj_list.append(cur_adj.squeeze())
                cur_adj = torch.stack(adj_list, -1)

            # KNN with eucledian distance
            elif self.args.graph_type == 2:
                cur_meta = cur_meta.cpu().numpy()
                cur_meta = np.nan_to_num(cur_meta, nan=0)
                graph_row_neighbors = utils.get_k_nearest_neighbors(
                    cur_meta, 10)
                cur_adj = utils.get_adjacency_matrix(cur_meta,
                                                     graph_row_neighbors)
                cur_adj = torch.tensor(cur_adj)
                cur_adj = torch.eye(cur_adj.shape[0]) + cur_adj.triu(1) + \
                          cur_adj.triu(1).t().contiguous()
                cur_adj = cur_adj.long()

            # HEOM (https://github.com/KacperKubara/distython)
            elif self.args.graph_type == 3:
                raise ValueError(f"graph_type = {self.args.graph_type} "
                                 f"currently not supported")

            elif self.args.graph_type == 4:
                raise ValueError(f"graph_type = {self.args.graph_type} "
                                 f"currently not supported")

            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj

    def drop_one_elem_columns(self, df):
        """Drop columns in array if there is only a single unique element in
        it plus np.nan."""
        df_ = df.copy()

        # Incldue  columns in dataframe
        include_idx = []
        for i in df_.columns:
            len_unique = df_[i].dropna().unique().size
            if len_unique > 1:
                include_idx.append(i)

        df_ = df_[include_idx]
        return df_


class TADPOLEDataset(DataProvider):
    """TADPOLE dataset provider."""
    def __init__(self, args):
        super(TADPOLEDataset, self).__init__()
        self.args = args
        self.all_non_numerical_idx = None
        self._load_data()

    def _load_data(self):
        """Load data x and data y."""

        path_data_x = '../data/tadpole/adni_one_baseline_feature_data.csv'
        path_data_y = '../data/tadpole/adni_one_baseline_label_data.csv'
        path_meta = '../data/tadpole/adni_one_baseline_meta_data.csv'
        read_data_x = pd.read_csv(path_data_x)
        read_data_y = pd.read_csv(path_data_y)  # 0 NL, 1, MCI, 2 Dementia
        read_data_meta = pd.read_csv(path_meta)[['AGE', 'PTGENDER', 'APOE4']]

        # Replace gender to numeric
        read_data_meta.PTGENDER = read_data_meta.PTGENDER.replace('Male', 0)
        read_data_meta.PTGENDER = read_data_meta.PTGENDER.replace('Female', 1)

        new_data_x = np.array(read_data_x).astype(np.float32)
        new_data_y = np.array(read_data_y).astype(np.float32)
        new_data_meta = np.array(read_data_meta).astype(np.float32)

        # Concat meta-information with feature vector input
        concat_meta = pd.DataFrame(new_data_meta)
        concat_meta.iloc[:, 2] = concat_meta.iloc[:, 2].replace(0, 'zero')
        concat_meta.iloc[:, 2] = concat_meta.iloc[:, 2].replace(1, 'one')
        concat_meta.iloc[:, 2] = concat_meta.iloc[:, 2].replace(2, 'two')
        concat_meta = concat_meta.to_numpy()
        new_data_x = np.concatenate([concat_meta, new_data_x], 1)
        print(new_data_x.shape, new_data_y.shape, new_data_meta.shape)

        self.data_x = new_data_x
        self.data_y = self.to_one_hot_encoding(new_data_y)
        self.numerical_idx = np.arange(new_data_x.shape[-1])
        self.numerical_idx = np.delete(self.numerical_idx, [2])  # Remove APOE column idx
        self.non_num_idx = np.array([2])
        self.all_non_numerical_idx = None

        # Calculate adjacency matrix
        self.meta_inf = new_data_meta.astype('float32')
        if self.args.graph_type:
            self.adj = self.get_adjacency()

    def get_adjacency(self):
        save_path = self.args.output_path

        save_path = os.path.join(save_path, 'adj_matrix_all.pt')

        if self.args.graph_type == 1:
            save_path = save_path.replace('.pt', '_thresholded.pt')
        elif self.args.graph_type == 2:
            save_path = save_path.replace('.pt', '_knn.pt')
        elif self.args.graph_type == 3:
            save_path = save_path.replace('.pt', '_heom.pt')
        elif self.args.graph_type == 4:
            save_path = save_path.replace('.pt', '_HeomSubset.pt')
        else:
            raise NotImplementedError

        # Load adjacency matrix if available, otherwise calculate it
        if os.path.exists(save_path):
            with open(save_path, 'rb') as fpath:
                cur_adj = torch.load(fpath)
        else:
            # Calculate adjacency using given similarity metric.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cur_meta = torch.from_numpy(self.meta_inf).float().to(device)
            # Thresholded similarity metric
            threshold_list = [5, 0, 0]
            if self.args.graph_type == 1:
                cur_adj = utils.get_thresholded_graph(cur_meta, threshold_list)
            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj


class NHANESDataset(DataProvider):
    """ NHANES dataset provider. """

    def __init__(self, args):
        self.args = args
        self.all_non_numerical_idx = None

        data_path = '../data/nhanes/'

        data_x_path = os.path.join(data_path, 'features_all_cols.npy')
        data_y_path = os.path.join(data_path, 'targets.npy')

        self.data_x = np.load(data_x_path)
        self.data_y = self.to_one_hot_encoding(np.load(data_y_path))
        self.numerical_idx = np.arange(self.data_x.shape[-1])
        self.non_num_idx = None
        self.all_non_numerical_idx = None

        # Calculate adjacency matrix
        self.meta_inf = self.data_x
        if self.args.graph_type:
            self.adj = self.get_adjacency()

    def get_adjacency(self):
        save_path = self.args.output_path

        save_path = os.path.join(save_path, 'adj_matrix_all.pt')

        if self.args.graph_type == 1:
            save_path = save_path.replace('.pt', '_thresholded.pt')
        else:
            raise NotImplementedError

        # Load adjacency matrix if available, otherwise calculate it
        if os.path.exists(save_path):
            with open(save_path, 'rb') as fpath:
                cur_adj = torch.load(fpath)
        else:
            # Calculate adjacency using given similarity metric.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cur_meta = torch.from_numpy(self.meta_inf).float().to(device)
            # Thresholded similarity metric
            cur_meta = cur_meta[:, :2]
            threshold_list = cur_meta.std(0)
            if self.args.graph_type == 1:
                cur_adj = utils.get_thresholded_graph(cur_meta, threshold_list)
            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj
