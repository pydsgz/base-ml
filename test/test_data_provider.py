import argparse
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import pprint
pprint.pprint(sys.path)
from base_ml import data_provider as dp

# class TestDataProvider:
#     def test_load_data(self):
#         assert False
#
#     def test_to_one_hot_encoding(self):
#         assert False
#
#     def test_set_binary_feature_idx_list(self):
#         assert False
#
#     def test_set_len_numerical_feat(self):
#         assert False
#
#     def test_dataset_provider_attributes(self):
#         assert False
#
#
# class TestTranductiveMGMCDataset:
#     assert False
#
#
# class TestTransductiveGNNDataset:
#     assert False
#
#
# class TestInductiveGNNDataset:
#     assert False
#
#
# class TestMNISTDataset:
#     assert False


def test_mnist_dataset():
    # parser = argparse.ArgumentParser(description='PyTorch Dizzyreg Experiments')
    # parser.add_argument('-x', '--exp_num', default=7, type=int, help='')
    # parser.add_argument('-d', '--dataset', default='dizzyreg3', type=str,
    #                     help='Dataset name can be dizzyreg{1,2,3}.')
    # parser.add_argument('--output_path', default='./outputs/%s/',
    #                     type=str,
    #                     help='Location where all outputs will be saved.')
    # parser.add_argument('--graph_type',
    #                     default=1,
    #                     type=int,
    #                     help='1 for thresholded, 2 for knn, 3 for HEOM')
    # parser.add_argument('--gnn_layers', default=[30, 12, 12, 2], type=int,
    #                     help='input, hidden, and output layer dimensions')
    # parser.add_argument('-hp', '--hp_iter', default=100,
    #                     type=int,
    #                     help='Number of iterations to run hyperparameter '
    #                          'search.')
    # parser.add_argument('-r', '--remainder', default='passthrough',
    #                     type=str,
    #                     help='"drop" or "passthrough". Passthrough will use '
    #                          'remaining column not specified in column '
    #                          'transformer. Drop will exclude remaining '
    #                          'columns.')
    # parser.add_argument('--device', default=0, type=int,
    #                     help='Index of gpu device to use.')
    # parser.add_argument('--save_embedding', default=1, type=int,
    #                     help='Set to 1 to save embedding.')
    # parser.add_argument('--p_remain', default=1.0, type=float,
    #                     help='If 1.0 will not simulate missingess, else if '
    #                          'between (0,1) will only retain given p_remain '
    #                          'ration of features and randomly simulate '
    #                          'missingness.')
    # args = parser.parse_args()
    #
    # # Parameters used during training for particular session
    # args.output_path = os.path.join(args.output_path, 'exp_%03d' % args.exp_num)
    # args.cv_type = 2  # 2 for Stratified
    # args.cv_folds = 10
    # args.rand_seed = 0
    # args.baseline_only = True
    # args.device = 2
    # args.device = 'cuda:%s' % args.device if torch.cuda.is_available() else \
    #     'cpu'
    # mlp_max_iter = 1000
    # args.output_path = args.output_path % args.dataset
    # if not os.path.exists(args.output_path):
    #     os.makedirs(args.output_path)
    #
    # # For reproducibility
    # np.random.seed(args.rand_seed)
    # torch.manual_seed(args.rand_seed)
    # random_state = np.random.RandomState(args.rand_seed)
    # args.random_state = random_state
    #
    # dataset = dp.MNISTDataset(args)
    assert 1 == 0
    #
    # print(dataset.data_x)
    # print(dataset.data_y)
    # print(dataset.numerical_idx)
    # print(dataset.non_num_idx)
    # print(dataset.adj)
    # print(dataset.meta_inf)
    # print(dataset.args)
    # print(dataset.binary_idx_list)
    # print(dataset.len_numerical_feat)
    # print(dataset.all_non_numerical_idx)

# class TestDizzyregDataset:
#     assert False
