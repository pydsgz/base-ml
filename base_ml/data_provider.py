import os
import torch

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from base_ml import utils
from sklearn.neighbors import NearestNeighbors
# from distython import HEOM
from torchvision import datasets, transforms


class DataProvider(Dataset):
    """Data provider for machine learning pipelines.

    Attributes:
        data_x: Input features of size n x f.
        data_y: Input target of size n x t.
        numerical_idx: List of indeces which are numerical features in data_x.
        non_num_idx: List of indeces which are non-numerical features in data_x.
        adj: Adjacency matrix of size n x n
        meta_inf: Meta-information of size n x m, this will be used to
            calculate the adjacency matrix.
        args: Argument parser object.
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
        """Convert target class labels to one-hot encoding"""
        target_data = target_data.squeeze()
        n_class = len(np.unique(target_data))
        res = np.eye(n_class)[target_data.astype(int)]
        return res

    def set_binary_feature_idx_list(self):
        self.binary_idx_list = np.array(utils.get_binary_indices(self.data_x, self.non_num_idx))
        self.non_num_idx = np.setdiff1d(self.non_num_idx, self.binary_idx_list)

    def set_len_numerical_feat(self):
        if self.binary_idx_list is not None:
            self.len_numerical_feat = len(self.numerical_idx) + len(self.binary_idx_list)
        else:
            self.len_numerical_feat = len(self.numerical_idx)


class TransductiveMGMCDataset(Dataset):
    """Dataset provider for MGMC learning"""
    def __init__(self, feat_data, adj_matrix, target_data, set_idx, val_idx=None, mask_matrix=None, is_training=None,
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
                target_for_concat[np.setdiff1d(np.arange(target_data.shape[0]), set_idx), :] = 0.0

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
            sim_missing_feat_data, mask_simulated = utils.select_p_data_df(cur_feat_matrix, p_select=data_remain,
                                                                           random_seed=0)
            mask_matrix = np.isnan(sim_missing_feat_data)
            sim_missing_feat_data = np.nan_to_num(sim_missing_feat_data, 0.0)
            feat_data = sim_missing_feat_data
            self.missing_idx_sim = mask_simulated

        # Combine feature matrix and label matrix
        self.feat_data = np.concatenate([feat_data, target_for_concat], -1).astype(np.float32)

        # Class weighting
        class_counts = pd.Series(target_data.argmax(-1)).value_counts()
        self.class_weight = np.array(class_counts.max() / class_counts)

        assert len(feat_data) == len(target_data)

    def __len__(self):
        return len(self.feat_data)

    def __getitem__(self, idx):
        # ret_val = {'feat_data': self.feat_data[idx],
        #            'adj_matrix': self.adj_mat[idx],
        #            'idx': idx,
        #            'tr_idx': self.idx,
        #            'val_idx': self.val_idx,
        #            'mask_matrix': self.mask_matrix[idx]
        #            }
        # return ret_val, self.target_data[idx]
        return (self.feat_data[idx], self.adj_mat[idx], idx, self.idx, self.val_idx, self.mask_matrix[idx]), \
               self.target_data[idx]


class TransductiveGNNDataset(Dataset):
    """Dataset provider for Transductive Classification using Graph Neural Networks."""
    def __init__(self, feat_data, adj_matrix, target_data, set_idx, val_idx=None, mask_matrix=None):

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


class VO2PeakDatasetReg(DataProvider):
    """VO2Peak regression dataset."""
    def __init__(self, args):
        super(VO2PeakDatasetReg, self).__init__()
        self.args = args
        self._load_data()

    def _load_data(self):
        """Load data x and data y."""
        data_path = '../data/chd/CHD_Camp.xlsx'
        read_data_x = pd.read_excel(data_path)

        # Sort
        read_data_x = read_data_x.sort_values(['Person_ID', 'age'])

        # Drop multiple
        if self.args.baseline_only:
            read_data_x = read_data_x.drop_duplicates(['Person_ID'],
                                                      keep='first')
            read_data_x = read_data_x.reset_index(drop=True)

        # Drop age < 14 years old
        read_data_x = read_data_x[list(read_data_x.age >= 14)]
        read_data_x = read_data_x.reset_index(drop=True)

        col_of_interest = ["Diagnose", "severity", "age", "sex", "weight",
                           "height", "surgery", "Pacer", "BBlocker",
                           "CaAntag", "Digoxin", "ACEI", "SpO2_rest",
                           "SF_htran", "physical_sum_score", "mental_sum_score"]

        data_x = np.array(read_data_x[col_of_interest])
        data_vo2peak = np.array(read_data_x[['VO2peakre']]).ravel()
        data_vo2peak = data_vo2peak.reshape(-1, 1)
        data_vo2peak = data_vo2peak.astype(np.float32)

        # one hot encoding diagnosis
        cur_data_x = read_data_x[col_of_interest]

        # Replace medication to 0
        new_medication_df = cur_data_x[
            ['surgery', 'Pacer', 'BBlocker', 'CaAntag', 'Digoxin',
             'ACEI']].fillna(0)

        # Replace zero to nan spo2rest
        new_spo2_rest = cur_data_x.SpO2_rest.copy()
        new_spo2_rest[new_spo2_rest == 0.0] = np.nan
        new_spo2_rest = pd.DataFrame(new_spo2_rest)

        # SF_htran -> rescale [-1,1]
        new_sf_htran = cur_data_x.SF_htran.copy()
        new_sf_htran[new_sf_htran == 0] = -1.0
        new_sf_htran[new_sf_htran == 25.] = -0.5
        new_sf_htran[new_sf_htran == 50.] = 0.0
        new_sf_htran[new_sf_htran == 75] = 0.5
        new_sf_htran[new_sf_htran == 100.] = 1.0

        # Columns from original dataframes
        old_df = cur_data_x[
            ["Diagnose", 'severity', 'age', 'sex', 'weight', 'height',
             'physical_sum_score', 'mental_sum_score']]

        # Combine all dataframes
        final_data_df = pd.concat(
            [old_df, new_medication_df, new_sf_htran, new_spo2_rest], 1)

        # Define which are numeric and non-numeric features for pre-processing
        numeric_features = ['age', 'weight', 'height', 'physical_sum_score',
                            'mental_sum_score', 'SF_htran', 'SpO2_rest']
        is_numeric = final_data_df.columns.isin(numeric_features)
        numeric_features = is_numeric.nonzero()[0]
        categorical_features = ['Diagnose', 'severity', 'sex']
        is_cat = final_data_df.columns.isin(categorical_features)
        categorical_features = is_cat.nonzero()[0]

        # Change categorical features to object type as numerical type does
        # not work with SimpleImputer when categorical features are encoded
        # as numerical.
        final_data_df.iloc[:, categorical_features] = \
            final_data_df.iloc[:, categorical_features].astype(str)

        # data_x = np.array(final_data_df)
        self.data_x = np.array(final_data_df)
        self.data_y = data_vo2peak
        self.numerical_idx = numeric_features
        self.non_num_idx = categorical_features


class VECO2SlopeReg(DataProvider):
    """VO2Peak regression dataset."""
    def __init__(self, args):
        super(VECO2SlopeReg, self).__init__()
        self.args = args
        self._load_data()

    def _load_data(self):
        """Load data x and data y."""
        data_path = '../data/chd/CHD_Camp.xlsx'
        read_data_x = pd.read_excel(data_path)

        # Sort
        read_data_x = read_data_x.sort_values(['Person_ID', 'age'])

        # Drop multiple
        if self.args.baseline_only:
            read_data_x = read_data_x.drop_duplicates(['Person_ID'],
                                                      keep='first')
            read_data_x = read_data_x.reset_index(drop=True)

        # Drop age < 14 years old
        read_data_x = read_data_x[list(read_data_x.age >= 14)]
        read_data_x = read_data_x.reset_index(drop=True)

        col_of_interest = ["Diagnose", "severity", "age", "sex", "weight",
                           "height", "surgery", "Pacer", "BBlocker",
                           "CaAntag", "Digoxin", "ACEI", "SpO2_rest",
                           "SF_htran", "physical_sum_score", "mental_sum_score"]

        # one hot encoding diagnosis
        cur_data_x = read_data_x[col_of_interest]

        # Replace medication to 0
        new_medication_df = cur_data_x[
            ['surgery', 'Pacer', 'BBlocker', 'CaAntag', 'Digoxin',
             'ACEI']].fillna(0)

        # Replace zero to nan spo2rest
        new_spo2_rest = cur_data_x.SpO2_rest.copy()
        new_spo2_rest[new_spo2_rest == 0.0] = np.nan
        new_spo2_rest = pd.DataFrame(new_spo2_rest)

        # SF_htran -> rescale [-1,1]
        new_sf_htran = cur_data_x.SF_htran.copy()
        new_sf_htran[new_sf_htran == 0] = -1.0
        new_sf_htran[new_sf_htran == 25.] = -0.5
        new_sf_htran[new_sf_htran == 50.] = 0.0
        new_sf_htran[new_sf_htran == 75] = 0.5
        new_sf_htran[new_sf_htran == 100.] = 1.0

        # Columns from original dataframes
        old_df = cur_data_x[
            ["Diagnose", 'severity', 'age', 'sex', 'weight', 'height',
             'physical_sum_score', 'mental_sum_score']]

        # Combine all dataframes
        final_data_df = pd.concat(
            [old_df, new_medication_df, new_sf_htran, new_spo2_rest], 1)

        # Load VECO2slope data
        data_veco2slope = read_data_x[['VECO2slope']]
        # Drop rows with missing VECO2slope data
        mask_ = np.array(data_veco2slope.notna())
        new_data_veco2slope = np.array(data_veco2slope[mask_])
        new_data_veco2slope = new_data_veco2slope.astype(np.float32)
        new_data_x = final_data_df[mask_]

        # Drop rows with SpO2_rest > 95
        cur_spo2_res = read_data_x[['SpO2_rest']][mask_]
        spo2_mask = list(cur_spo2_res.SpO2_rest > 95)
        new_data_x = new_data_x[spo2_mask]
        new_data_veco2slope = new_data_veco2slope[spo2_mask]

        # Define which are numeric and non-numeric features for pre-processing
        numeric_features = ['age', 'weight', 'height', 'physical_sum_score',
                            'mental_sum_score', 'SF_htran', 'SpO2_rest']
        is_numeric = final_data_df.columns.isin(numeric_features)
        numeric_features = is_numeric.nonzero()[0]
        categorical_features = ['Diagnose', 'severity', 'sex']
        is_cat = final_data_df.columns.isin(categorical_features)
        categorical_features = is_cat.nonzero()[0]

        # Change categorical features to object type as numerical type does
        # not work with SimpleImputer when categorical features are encoded
        # as numerical.
        new_data_x.iloc[:, categorical_features] = \
            new_data_x.iloc[:, categorical_features].astype(str)

        # data_x = np.array(final_data_df)
        self.data_x = np.array(new_data_x)
        self.data_y = new_data_veco2slope
        self.numerical_idx = numeric_features
        self.non_num_idx = categorical_features


# class CHDThreeClassDataset(DataProvider):
#     """Three-class classification dataset."""
#     def __init__(self, args):
#         super(CHDThreeClassDataset, self).__init__()
#         self.args = args
#         self.all_non_numerical_idx = None
#         self._load_data()
#
#     def _load_data(self):
#         """Load data x and data y."""
#         data_path = '../data/chd/CHD_Camp.xlsx'
#         read_data_x = pd.read_excel(data_path)
#
#         # Sort
#         read_data_x = read_data_x.sort_values(['Person_ID', 'age'])
#
#         # Drop multiple
#         if self.args.baseline_only:
#             read_data_x = read_data_x.drop_duplicates(['Person_ID'],
#                                                       keep='first')
#             read_data_x = read_data_x.reset_index(drop=True)
#
#         # Drop age < 14 years old
#         read_data_x = read_data_x[list(read_data_x.age >= 14)]
#         read_data_x = read_data_x.reset_index(drop=True)
#
#         col_of_interest = ["Diagnose", "severity", "age", "sex", "weight",
#                            "height", "surgery", "Pacer", "BBlocker",
#                            "CaAntag", "Digoxin", "ACEI", "SpO2_rest",
#                            "SF_htran", "physical_sum_score", "mental_sum_score"]
#
#         data_x = np.array(read_data_x[col_of_interest])
#         data_vo2peak = np.array(read_data_x[['VO2peakre']]).ravel()
#         data_vo2peak = data_vo2peak.reshape(-1, 1)
#         data_vo2peak = data_vo2peak.astype(np.float32)
#
#         # one hot encoding diagnosis
#         cur_data_x = read_data_x[col_of_interest]
#
#         # Replace medication to 0
#         new_medication_df = cur_data_x[
#             ['surgery', 'Pacer', 'BBlocker', 'CaAntag', 'Digoxin',
#              'ACEI']].fillna(0)
#
#         # Replace zero to nan spo2rest
#         new_spo2_rest = cur_data_x.SpO2_rest.copy()
#         new_spo2_rest[new_spo2_rest == 0.0] = np.nan
#         new_spo2_rest = pd.DataFrame(new_spo2_rest)
#
#         # SF_htran -> rescale [-1,1]
#         new_sf_htran = cur_data_x.SF_htran.copy()
#         new_sf_htran[new_sf_htran == 0] = -1.0
#         new_sf_htran[new_sf_htran == 25.] = -0.5
#         new_sf_htran[new_sf_htran == 50.] = 0.0
#         new_sf_htran[new_sf_htran == 75] = 0.5
#         new_sf_htran[new_sf_htran == 100.] = 1.0
#
#         # Columns from original dataframes
#         old_df = cur_data_x[
#             ["Diagnose", 'severity', 'age', 'sex', 'weight', 'height',
#              'physical_sum_score', 'mental_sum_score']]
#
#         # Combine all dataframes
#         final_data_df = pd.concat(
#             [old_df, new_medication_df, new_sf_htran, new_spo2_rest], 1)
#
#         # Define which are numeric and non-numeric features for pre-processing
#         numeric_features = ['age', 'weight', 'height', 'physical_sum_score',
#                             'mental_sum_score', 'SF_htran', 'SpO2_rest']
#         is_numeric = final_data_df.columns.isin(numeric_features)
#         numeric_features = is_numeric.nonzero()[0]
#         categorical_features = ['Diagnose', 'severity']
#         is_cat = final_data_df.columns.isin(categorical_features)
#         categorical_features = is_cat.nonzero()[0]
#
#         # all non-numerical columns for HEOM
#         all_non_numerical = ["Diagnose", "severity", "sex", "surgery",
#                              "Pacer", "BBlocker", "CaAntag", "Digoxin",
#                              "ACEI"]
#         all_non_numerical = final_data_df.columns.isin(all_non_numerical)
#         self.all_non_numerical_idx = all_non_numerical.nonzero()[0]
#
#         # Change categorical features to object type as numerical type does
#         # not work with SimpleImputer when categorical features are encoded
#         # as numerical.
#         final_data_df.iloc[:, categorical_features] = \
#             final_data_df.iloc[:, categorical_features].astype(str)
#
#         # Class labels
#         dum_y = read_data_x[['age', 'VO2peakre', 'risk_classes']]
#         dum_y['new_class'] = 0
#         dum_y.new_class[(dum_y.VO2peakre <= 60)] = 3
#         dum_y.new_class[(dum_y.VO2peakre > 80)] = 1
#         dum_y.new_class[(dum_y.VO2peakre > 60) & (dum_y.VO2peakre <= 80) & (
#                     dum_y.age <= 80)] = 2
#         y_data_classification = np.array(dum_y.new_class).astype('int64') - 1
#         print((dum_y.new_class - 1).value_counts())
#         # data_x = np.array(final_data_df)
#         self.data_x = np.array(final_data_df)
#         self.data_y = y_data_classification
#         self.numerical_idx = numeric_features
#         self.non_num_idx = categorical_features
#
#         # Calculate adjacency matrix
#         self.meta_inf = self.data_x.astype('float32')[[6, 10, 4]]
#         if self.args.graph_type:
#             self.adj = self.get_adjacency(data_path)
#
#         class_counts = pd.Series(self.data_y).value_counts().sort_index()
#         self.class_weighting = torch.tensor(class_counts.max() / class_counts)
#         self.class_weighting = self.class_weighting / self.class_weighting.sum()
#
#
#     def get_adjacency(self, data_path=None):
#         save_path = os.path.dirname(data_path)
#         if self.args.baseline_only:
#             save_path = os.path.join(save_path, 'adj_matrix_baseline.pt')
#         else:
#             save_path = os.path.join(save_path, 'adj_matrix_all.pt')
#
#         if self.args.graph_type == 1:
#             save_path = save_path.replace('.pt', '_thresholded.pt')
#         elif self.args.graph_type == 2:
#             save_path = save_path.replace('.pt', '_knn.pt')
#         elif self.args.graph_type == 3:
#             save_path = save_path.replace('.pt', '_heom.pt')
#         else:
#             raise NotImplementedError
#
#         # Load adjacency matrix if available, otherwise calculate it
#         if os.path.exists(save_path):
#             with open(save_path, 'rb') as fpath:
#                 device = "cuda" if torch.cuda.is_available() else "cpu"
#                 cur_adj = torch.load(fpath)
#         else:
#             # Calculate adjacency using given similarity metric.
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             cur_meta = torch.from_numpy(self.meta_inf).float().to(device)
#             # Thresholded gaussian
#             if self.args.graph_type == 1:
#                 dist = cur_meta[:, None, :] - cur_meta[None, :, :]
#                 dist = torch.sum(dist, dim=2)
#                 dist = torch.sqrt(dist ** 2)
#                 cur_adj = dist < 5
#                 # cur_adj = torch.exp(-dist)
#                 # cur_adj = cur_adj > 0.5
#                 cur_adj = cur_adj.long()
#
#             # KNN with eucledian distance
#             elif self.args.graph_type == 2:
#                 cur_meta = cur_meta.cpu().numpy()
#                 cur_meta = np.nan_to_num(cur_meta, nan=0)
#                 graph_row_neighbors = utils.get_k_nearest_neighbors(
#                     cur_meta, 10)
#                 cur_adj = utils.get_adjacency_matrix(cur_meta,
#                                                      graph_row_neighbors)
#                 cur_adj = torch.tensor(cur_adj)
#                 cur_adj = torch.eye(cur_adj.shape[0]) + cur_adj.triu(1) + \
#                           cur_adj.triu(1).t().contiguous()
#                 cur_adj = cur_adj.long()
#
#             # HEOM (https://github.com/KacperKubara/distython)
#             elif self.args.graph_type == 3:
#                 cur_meta = cur_meta.cpu().numpy()
#                 cur_meta = np.nan_to_num(cur_meta, nan=-99999)
#                 heom_metric = HEOM(cur_meta, self.all_non_numerical_idx,
#                                    nan_equivalents=[-99999])
#                 graph_row_neighbors = utils.get_k_nearest_neighbors(
#                     cur_meta, 10, metric=heom_metric.heom)
#                 cur_adj = utils.get_adjacency_matrix(cur_meta,
#                                                      graph_row_neighbors)
#                 cur_adj = torch.tensor(cur_adj)
#                 cur_adj = torch.eye(cur_adj.shape[0]) + cur_adj.triu(1) + \
#                           cur_adj.triu(1).t().contiguous()
#                 cur_adj = cur_adj.long()
#             else:
#                 raise NotImplementedError
#             with open(save_path, 'wb') as fpath:
#                 torch.save(cur_adj, fpath)
#         return cur_adj


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
                cur_adj = utils.get_thresholded_eucledian(data_arr, threshold_list)
            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj


# class CHDTwoClassDataset(DataProvider):
#     """Two-class classification dataset."""
#     def __init__(self, args):
#         super(CHDTwoClassDataset, self).__init__()
#         self.args = args
#         self.all_non_numerical_idx = None
#         self._load_data()
#
#     def _load_data(self):
#         """Load data x and data y."""
#         data_path = '../data/chd/CHD_Camp.xlsx'
#         read_data_x = pd.read_excel(data_path)
#
#         # Sort
#         read_data_x = read_data_x.sort_values(['Person_ID', 'age'])
#
#         # Drop multiple
#         if self.args.baseline_only:
#             read_data_x = read_data_x.drop_duplicates(['Person_ID'],
#                                                       keep='first')
#             read_data_x = read_data_x.reset_index(drop=True)
#
#         # Drop age < 14 years old
#         read_data_x = read_data_x[list(read_data_x.age >= 14)]
#         read_data_x = read_data_x.reset_index(drop=True)
#
#         col_of_interest = ["Diagnose", "severity", "age", "sex", "weight",
#                            "height", "surgery", "Pacer", "BBlocker",
#                            "CaAntag", "Digoxin", "ACEI", "SpO2_rest",
#                            "SF_htran", "physical_sum_score", "mental_sum_score"]
#
#         data_x = np.array(read_data_x[col_of_interest])
#         data_vo2peak = np.array(read_data_x[['VO2peakre']]).ravel()
#         data_vo2peak = data_vo2peak.reshape(-1, 1)
#         data_vo2peak = data_vo2peak.astype(np.float32)
#
#         # one hot encoding diagnosis
#         cur_data_x = read_data_x[col_of_interest]
#
#         # Replace medication to 0
#         new_medication_df = cur_data_x[
#             ['surgery', 'Pacer', 'BBlocker', 'CaAntag', 'Digoxin',
#              'ACEI']].fillna(0)
#
#         # Replace zero to nan spo2rest
#         new_spo2_rest = cur_data_x.SpO2_rest.copy()
#         new_spo2_rest[new_spo2_rest == 0.0] = np.nan
#         new_spo2_rest = pd.DataFrame(new_spo2_rest)
#
#         # SF_htran -> rescale [-1,1]
#         new_sf_htran = cur_data_x.SF_htran.copy()
#         new_sf_htran[new_sf_htran == 0] = -1.0
#         new_sf_htran[new_sf_htran == 25.] = -0.5
#         new_sf_htran[new_sf_htran == 50.] = 0.0
#         new_sf_htran[new_sf_htran == 75] = 0.5
#         new_sf_htran[new_sf_htran == 100.] = 1.0
#
#         # Columns from original dataframes
#         old_df = cur_data_x[
#             ["Diagnose", 'severity', 'age', 'sex', 'weight', 'height',
#              'physical_sum_score', 'mental_sum_score']]
#
#         # Combine all dataframes
#         final_data_df = pd.concat(
#             [old_df, new_medication_df, new_sf_htran, new_spo2_rest], 1)
#
#         # Define which are numeric and non-numeric features for pre-processing
#         numeric_features = ['age', 'weight', 'height', 'physical_sum_score',
#                             'mental_sum_score', 'SF_htran', 'SpO2_rest']
#         is_numeric = final_data_df.columns.isin(numeric_features)
#         numeric_features = is_numeric.nonzero()[0]
#         categorical_features = ['Diagnose', 'severity']
#         is_cat = final_data_df.columns.isin(categorical_features)
#         categorical_features = is_cat.nonzero()[0]
#
#         # all non-numerical columns for HEOM
#         all_non_numerical = ["Diagnose", "severity", "sex", "surgery",
#                              "Pacer", "BBlocker", "CaAntag", "Digoxin",
#                              "ACEI"]
#         all_non_numerical = final_data_df.columns.isin(all_non_numerical)
#         self.all_non_numerical_idx = all_non_numerical.nonzero()[0]
#
#         # Change categorical features to object type as numerical type does
#         # not work with SimpleImputer when categorical features are encoded
#         # as numerical.
#         final_data_df.iloc[:, categorical_features] = \
#             final_data_df.iloc[:, categorical_features].astype(str)
#
#         # Class labels
#         dum_y = read_data_x[['age', 'VO2peakre', 'risk_classes']]
#         dum_y['new_class'] = 0
#         dum_y.new_class[(dum_y.VO2peakre <= 60)] = 3
#         dum_y.new_class[(dum_y.VO2peakre > 80)] = 1
#         dum_y.new_class[(dum_y.VO2peakre > 60) & (dum_y.VO2peakre <= 80) & (
#                     dum_y.age <= 80)] = 2
#         print(dum_y.new_class.value_counts())
#         y_data_classification = np.array(dum_y.new_class) - 1
#         y_data_classification[y_data_classification == 2] = 1
#         y_data_classification = y_data_classification.astype('int64')
#
#         # data_x = np.array(final_data_df)
#         self.data_x = np.array(final_data_df)
#         self.data_y = y_data_classification
#         self.numerical_idx = numeric_features
#         self.non_num_idx = categorical_features
#
#         # Calculate adjacency matrix
#         self.meta_inf = self.data_x.astype('float32')
#         if self.args.graph_type:
#             self.adj = self.get_adjacency(data_path)
#
#     def get_adjacency(self, data_path=None):
#         save_path = os.path.dirname(data_path)
#         if self.args.baseline_only:
#             save_path = os.path.join(save_path, 'adj_matrix_baseline.pt')
#         else:
#             save_path = os.path.join(save_path, 'adj_matrix_all.pt')
#
#         if self.args.graph_type == 1:
#             save_path = save_path.replace('.pt', '_thresholded.pt')
#         elif self.args.graph_type == 2:
#             save_path = save_path.replace('.pt', '_knn.pt')
#         elif self.args.graph_type == 3:
#             save_path = save_path.replace('.pt', '_heom.pt')
#         elif self.args.graph_type == 4:
#             save_path = save_path.replace('.pt', '_HeomSubset.pt')
#         else:
#             raise NotImplementedError
#
#         # Load adjacency matrix if available, otherwise calculate it
#         if os.path.exists(save_path):
#             with open(save_path, 'rb') as fpath:
#                 cur_adj = torch.load(fpath)
#         else:
#             # Calculate adjacency using given similarity metric.
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             cur_meta = torch.from_numpy(self.meta_inf).float().to(device)
#             # Thresholded gaussian
#             if self.args.graph_type == 1:
#                 dist = cur_meta[:, None, :] - cur_meta[None, :, :]
#                 dist = torch.sum(dist, dim=2)
#                 dist = torch.sqrt(dist ** 2)
#                 cur_adj = dist < 10
#                 # cur_adj = torch.exp(-dist)
#                 # cur_adj = cur_adj > 0.5
#                 cur_adj = cur_adj.long()
#
#             # KNN with eucledian distance
#             elif self.args.graph_type == 2:
#                 cur_meta = cur_meta.cpu().numpy()
#                 cur_meta = np.nan_to_num(cur_meta, nan=0)
#                 graph_row_neighbors = utils.get_k_nearest_neighbors(
#                     cur_meta, 10)
#                 cur_adj = utils.get_adjacency_matrix(cur_meta,
#                                                      graph_row_neighbors)
#                 cur_adj = torch.tensor(cur_adj)
#                 cur_adj = torch.eye(cur_adj.shape[0]) + cur_adj.triu(1) + \
#                           cur_adj.triu(1).t().contiguous()
#                 cur_adj = cur_adj.long()
#
#             # HEOM (https://github.com/KacperKubara/distython)
#             elif self.args.graph_type == 3:
#                 cur_meta = cur_meta.cpu().numpy()
#                 cur_meta = np.nan_to_num(cur_meta, nan=-99999)
#                 heom_metric = HEOM(cur_meta, self.all_non_numerical_idx,
#                                    nan_equivalents=[-99999])
#                 graph_row_neighbors = utils.get_k_nearest_neighbors(
#                     cur_meta, 10, metric=heom_metric.heom)
#                 cur_adj = utils.get_adjacency_matrix(cur_meta,
#                                                      graph_row_neighbors)
#                 cur_adj = torch.tensor(cur_adj)
#                 cur_adj = torch.eye(cur_adj.shape[0]) + cur_adj.triu(1) + \
#                           cur_adj.triu(1).t().contiguous()
#                 cur_adj = cur_adj.long()
#
#             # Use
#             elif self.args.graph_type == 4:
#                 cur_meta = cur_meta.cpu().numpy()
#                 cur_meta = np.nan_to_num(cur_meta, nan=-99999)
#                 heom_metric = HEOM(cur_meta, self.all_non_numerical_idx,
#                                    nan_equivalents=[-99999])
#                 graph_row_neighbors = utils.get_k_nearest_neighbors(
#                     cur_meta, 10, metric=heom_metric.heom)
#                 cur_adj = utils.get_adjacency_matrix(cur_meta,
#                                                      graph_row_neighbors)
#                 cur_adj = torch.tensor(cur_adj)
#                 cur_adj = torch.eye(cur_adj.shape[0]) + cur_adj.triu(1) + \
#                           cur_adj.triu(1).t().contiguous()
#                 cur_adj = cur_adj.long()
#             else:
#                 raise NotImplementedError
#             with open(save_path, 'wb') as fpath:
#                 torch.save(cur_adj, fpath)
#         return cur_adj


class GaitDataset(Dataset):
    """ Gait dataset loader. """
    def __init__(self, args):
        data_path = '../data/gait/gait_data_selected_2019.xlsx'
        read_data = pd.read_excel(data_path)

        # Merge new hopkins binary labels
        new_hopkins_label_path = \
            '../data/gait/SPSS_All_data_07102019.xlsx_Ahmad.xlsx'
        read_hopkins = pd.read_excel(new_hopkins_label_path).iloc[:, 1:2]
        read_hopkins = read_hopkins.replace(1, 0)
        read_hopkins = read_hopkins.replace(2, 1)
        # Load data_x and data_y
        data_x = read_data.iloc[:, :-4]
        data_x = data_x.iloc[:, 1:]

        all_data_y = read_data[['prosp_falls_yes_no',
                                'prosp_frequent_yes_no']]

        # 'prosp_Hopkins_dichotom_12vs34'
        all_data_y = pd.concat([all_data_y, read_hopkins], 1)

        # Preprocess data
        # Convert gender M to 0 Females to 1
        data_x.gender = data_x.gender.replace('M', 0)
        data_x.gender = data_x.gender.replace('F', 1)

        # Convert groups to one-hot-encoding
        enc = OneHotEncoder(handle_unknown='ignore')
        new_groups = enc.fit_transform(data_x.group.values.reshape(-1, 1))
        new_groups = new_groups.toarray()
        new_groups = pd.DataFrame(new_groups, columns=['group_%s' % x for x in
                                                       np.arange(1, 8)])
        data_x = data_x.drop(columns=['group'])
        data_x = pd.concat([data_x, new_groups], sort=False, axis=1)

        # 99 was use as a code for missing data for some columns
        data_x.replace(99, np.nan, inplace=True)

        print("data_x shape -> %s, all_data_y shape -> %s" % (
        data_x.shape, all_data_y.shape))

        if args.dataset[-1] == '1':
            data_y = all_data_y[['prosp_falls_yes_no']]
        elif args.dataset[-1] == '2':
            data_y = all_data_y[['prosp_frequent_yes_no']]
        elif args.dataset[-1] == '3':
            data_y = all_data_y[['prosp_Hopkins_dichotom_12vs34']]
        else:
            raise NotImplementedError('Unknown class index')

        # Drop rows with missing labels
        mask_1 = data_y.notna().values
        data_x = data_x[mask_1]
        data_y = data_y[mask_1]
        data_x = np.array(data_x)
        data_y = np.array(data_y)

        numeric_features = ['gender', 'group']
        is_numeric = read_data.columns.isin(numeric_features)
        numeric_features = is_numeric.nonzero()[0]

        read_data.columns

        numeric_features = None
        categorical_features = None
        self.data_x = np.array(data_x)
        self.data_y = np.array(data_y)
        self.numerical_idx = numeric_features
        self.non_num_idx = categorical_features

        # Calculate adjacency matrix
        self.meta_inf = self.data_x.astype('float32')
        if self.args.graph_type:
            self.adj = self.get_adjacency(data_path)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx], self.data_meta[idx]


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
                cur_meta = cur_meta.cpu().numpy()
                cur_meta = np.nan_to_num(cur_meta, nan=-99999)
                heom_metric = HEOM(cur_meta, self.all_non_numerical_idx,
                                   nan_equivalents=[-99999])
                graph_row_neighbors = utils.get_k_nearest_neighbors(
                    cur_meta, 10, metric=heom_metric.heom)
                cur_adj = utils.get_adjacency_matrix(cur_meta,
                                                     graph_row_neighbors)
                cur_adj = torch.tensor(cur_adj)
                cur_adj = torch.eye(cur_adj.shape[0]) + cur_adj.triu(1) + \
                          cur_adj.triu(1).t().contiguous()
                cur_adj = cur_adj.long()

            # Use
            elif self.args.graph_type == 4:
                cur_meta = cur_meta.cpu().numpy()
                cur_meta = np.nan_to_num(cur_meta, nan=-99999)
                heom_metric = HEOM(cur_meta, self.all_non_numerical_idx,
                                   nan_equivalents=[-99999])
                graph_row_neighbors = utils.get_k_nearest_neighbors(
                    cur_meta, 10, metric=heom_metric.heom)
                cur_adj = utils.get_adjacency_matrix(cur_meta,
                                                     graph_row_neighbors)
                cur_adj = torch.tensor(cur_adj)
                cur_adj = torch.eye(cur_adj.shape[0]) + cur_adj.triu(1) + \
                          cur_adj.triu(1).t().contiguous()
                cur_adj = cur_adj.long()
            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj

    def drop_one_elem_columns(self, df):
        """Drop columns in array if there is only a single unique element in it plus np.nan."""
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


class UKBBDataset(DataProvider):
    """UKBB dataset provider."""
    def __init__(self, args):
        super(UKBBDataset, self).__init__()
        self.args = args
        self.all_non_numerical_idx = None
        self._load_data()

    def _load_data(self):
        """Load data x and data y."""

        path_data = '../data/ukbb/UKBB2_playground.csv'
        read_data_x = pd.read_csv(path_data)

        # Drop data containing missing target
        # read_data_x = read_data_x[read_data_x.iloc[:, 2].notna()]

        # Drop 21 subjects/rows which are >=70 years old.
        mask = list(read_data_x.iloc[:, 2] < 70)
        read_data_x = read_data_x[mask]
        cur_feat_data = read_data_x

        # Subsample 5000 rows without replacement as this will mainly be used for Transductive learning comparison
        # cur_feat_data = read_data_x.sample(5000, random_state=self.args.random_state, replace=False)
        # cur_target_data = cur_feat_data[['Sex']]

        # Age group as target
        age_df = cur_feat_data.iloc[:, 2].copy()
        age_df[(age_df >= 40.0) & (age_df < 50)] = 0
        age_df[(age_df >= 50.0) & (age_df < 60)] = 1
        age_df[(age_df >= 60.0) & (age_df < 70)] = 2
        age_df[(age_df >= 70.0) & (age_df < 80)] = 3
        age_df[(age_df >= 80.0) & (age_df < 90)] = 4
        age_df[age_df >= 90.0] = 5
        # age_df = age_df[age_df.isin([0, 1, 2])]
        num_class = len(age_df.unique())
        idx_list = age_df.astype(int).to_numpy()
        cur_target_data = np.eye(num_class)[idx_list]

        cur_feat_data = cur_feat_data.iloc[:, 3:]  # year of birth was removed so we start at index 3
        cur_meta_data = cur_feat_data

        cur_feat_data = np.array(cur_feat_data).astype(np.float32)
        cur_target_data = np.array(cur_target_data).astype(np.float32)
        cur_meta_data = np.array(cur_meta_data).astype(np.float32)

        # Make sure target data is one-hot encoded
        if cur_target_data.shape[-1] == 1:
            num_class = len(np.unique(cur_target_data))
            cur_target_data = np.eye(num_class)[cur_target_data.astype(int).reshape(-1)]
            cur_target_data = cur_target_data.astype('float32')

        print('feature_matrix shape =', cur_feat_data.shape,
              'target data shape =', cur_target_data.shape,
              'meta data shape =', cur_meta_data.shape)
        self.data_x = cur_feat_data
        self.data_y = cur_target_data
        self.numerical_idx = np.arange(cur_feat_data.shape[-1])
        self.non_num_idx = None
        self.all_non_numerical_idx = None

        # Calculate adjacency matrix
        self.meta_inf = cur_meta_data
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
            threshold_list = [0.02]
            if self.args.graph_type == 1:
                cur_adj = utils.get_thresholded_eucledian(data_arr, threshold_list)
            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj


class ThyroidDataset(DataProvider):
    """
    Thyroid dataset loader for for Thyroid disease classification.
    (https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/)
    """
    def __init__(self, args):
        super().__init__()

        # Column names in dataframe
        names = ['age',                         # continuous
                 'sex',                         # binary 0/1
                 'on thyroxine',                # binary 0/1
                 'query on thyroxine',          # binary 0/1
                 'on antithyroid medication',   # binary 0/1
                 'sick',                        # binary 0/1
                 'pregnant',                    # binary 0/1
                 'thyroid surgery',             # binary 0/1
                 'I131 treatment',              # binary 0/1
                 'query hypothyroid',           # binary 0/1
                 'query hyperthyroid',          # binary 0/1
                 'lithium',                     # binary 0/1
                 'goitre',                      # binary 0/1
                 'tumor',                       # binary 0/1
                 'hypopituitary',               # binary 0/1
                 'psych',                       # binary 0/1
                 'TSH',                         # continuous
                 'T3',                          # continuous
                 'TT4',                         # continuous
                 'T4U',                         # continuous
                 'FTI1',                        # continuous
                 'diagnoses']					# 3 classes: 1, 2, 3

        self.args = args
        tr_csv_path = '../data/thyroid/ann-train.csv'
        ts_csv_path = '../data/thyroid/ann-test.csv'
        tr_data = pd.read_csv(tr_csv_path, header=None)
        ts_data = pd.read_csv(ts_csv_path, header=None)
        data_df = pd.concat([tr_data, ts_data], 0)
        data_df = np.array(data_df)
        cur_feat_data = data_df[:, :-1]
        cur_target_data = data_df[:, -1]
        cur_meta_data = cur_feat_data

        cur_feat_data = np.array(cur_feat_data).astype(np.float32)
        cur_target_data = np.array(cur_target_data).astype(np.float32) - 1  # so that labels are 0, 1, 2
        cur_meta_data = np.array(cur_meta_data).astype(np.float32)

        ###
        self.data_x = cur_feat_data
        self.data_y = self.to_one_hot_encoding(cur_target_data)
        self.numerical_idx = np.arange(cur_feat_data.shape[-1])
        self.non_num_idx = None
        self.all_non_numerical_idx = None

        # Calculate adjacency matrix
        self.meta_inf = cur_meta_data
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
            threshold_list = [1.68]
            if self.args.graph_type == 1:
                cur_adj = utils.get_thresholded_eucledian(data_arr, threshold_list)
            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj


class PPMIDataset(DataProvider):
    """PPMI dataset provider."""
    def __init__(self, args):
        super(PPMIDataset, self).__init__()
        self.args = args
        self.all_non_numerical_idx = None
        self._load_data()

    def _load_data(self):
        """Load feature matrix and class label matrix."""

        # data_x, data_meta, data_y = gmc_load.quick_load_tadpole()
        data_x_path = '../data/PPMIdata/dim_reduction_output.txt'
        data_meta_path = '../data/PPMIdata/non_imaging_ppmi_data.xls'
        data_x = np.genfromtxt(data_x_path, dtype='float', delimiter=',')

        # Load meta-information
        # MOCA
        data_moca = pd.read_excel(data_meta_path, 'MOCA')
        data_moca = data_moca[['PATNO', 'EVENT_ID', 'MCATOT']]
        data_moca = data_moca[(data_moca.EVENT_ID == 'SC') | (
                data_moca.EVENT_ID == 'V01')]
        data_moca.drop_duplicates('PATNO', inplace=True)

        # UPDRS
        load_data_uppdrs = pd.read_excel(data_meta_path, 'UPDRS')
        updrs_cols = load_data_uppdrs.columns[8:41]
        uppdrs_score = load_data_uppdrs[updrs_cols].sum(1)
        load_data_uppdrs = load_data_uppdrs[['PATNO', 'EVENT_ID']]
        load_data_uppdrs['uppdrs_score'] = uppdrs_score
        data_uppdrs = load_data_uppdrs[(load_data_uppdrs.EVENT_ID == 'BL') | (
                load_data_uppdrs.EVENT_ID == 'V01')]
        data_uppdrs.sort_values(['PATNO', 'uppdrs_score'], inplace=True)
        data_uppdrs.drop_duplicates(['PATNO'], inplace=True)

        # Gender
        data_gender = pd.read_excel(data_meta_path, 'Gender_and_Age')
        data_gender = data_gender[['PATNO', 'GENDER']]

        # Age
        data_age = pd.read_excel(data_meta_path, 'Gender_and_Age')
        age_ = 2018 - data_age['BIRTHDT']
        # data_acq = pd.read_csv('../data/PPMIdata/Magnetic_Resonance_Imaging.csv')
        # data_age = data_age[['PATNO', 'EVENT_ID', 'MRIDT']]
        data_age = data_age[['PATNO']]
        data_age['age'] = age_

        # From label 2 to 1 and from label 1 to 0
        # 1 Means with Parkinson's disease, 0 normal.
        data_y = pd.read_csv('../data/PPMIdata/ppmi_labels.csv', names=[
            'PATNO', 'labels'])
        data_y.labels = data_y.labels - 1

        # Merge MOCA
        new_data_meta = data_y[['PATNO']]
        new_data_meta = new_data_meta.merge(data_moca[['PATNO', 'MCATOT']],
                                            on=['PATNO'],
                                            how='left')

        # Merge UPPDRS score
        new_data_meta = new_data_meta.merge(data_uppdrs[['PATNO',
                                                         'uppdrs_score']],
                                            on='PATNO',
                                            how='left')
        # Use screening (SC) UPPDRS score of missing patient without BL/V01
        # UPPDRS score
        missing_id = list(new_data_meta.PATNO[
                              new_data_meta.uppdrs_score.isna()])
        include_uppdrs = load_data_uppdrs[load_data_uppdrs.PATNO.isin(
            missing_id)]
        # PATNO [60070, 3801, 3837, 4060, 4069, 3833]
        new_data_meta = new_data_meta.merge(include_uppdrs[['PATNO',
                                                            'uppdrs_score']],
                                            on=['PATNO'],
                                            how='left')
        new_uppdrs_score = new_data_meta.uppdrs_score_x.combine_first(
            new_data_meta.uppdrs_score_y)
        new_data_meta['uppdrs_score'] = new_uppdrs_score
        new_data_meta = new_data_meta[['PATNO', 'MCATOT', 'uppdrs_score']]

        # Merge age
        new_data_meta = new_data_meta.merge(data_age,
                                            on='PATNO',
                                            how='left')

        # Merge gender
        new_data_meta = new_data_meta.merge(data_gender,
                                            on='PATNO',
                                            how='left')

        # Remove PATNO column and rename columns
        new_data_meta.drop(columns=['PATNO'], inplace=True)
        new_data_meta.columns = ['MCATOT', 'UPPDRS', 'AGE', 'GENDER']

        # Meta informaton to use to build the graphs
        data_meta = new_data_meta

        # Drop PATNO column
        data_y.drop(columns=['PATNO'], inplace=True)

        new_data_x = np.array(data_x).astype(np.float32)
        new_data_y = np.array(data_y).astype(np.float32)
        new_data_meta = np.array(data_meta).astype(np.float32)

        print(new_data_x.shape, new_data_y.shape, new_data_meta.shape)

        self.data_x = new_data_x
        self.data_y = self.to_one_hot_encoding(new_data_y)
        self.numerical_idx = np.arange(new_data_x.shape[-1])
        self.non_num_idx = None

        # Calculate adjacency matrix
        self.meta_inf = new_data_meta.astype('float32')
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
            threshold_list = [0, 0, 1, 0]
            if self.args.graph_type == 1:
                cur_adj = utils.get_thresholded_graph(cur_meta, threshold_list)
            else:
                raise NotImplementedError
            with open(save_path, 'wb') as fpath:
                torch.save(cur_adj, fpath)
        return cur_adj
