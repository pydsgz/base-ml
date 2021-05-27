import argparse
import os
import numpy as np
import sys
import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression

import skorch
import torch
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from skopt.learning.gaussian_process.kernels import RBF
from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler
from skorch.dataset import CVSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
import hyperopt

sys.path.append('../')
from base_ml import models, utils, model_wrapper
from base_ml import data_provider as dp
from base_ml import trainer_provider as tp
from base_ml import hyperparam_provider as hparam


def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='PyTorch Dizzyreg Experiments')
    parser.add_argument('-x', '--exp_num', default=8, type=int, help='')
    parser.add_argument('-d', '--dataset', default='synthetic', type=str,
                        help='Dataset name can be dizzyreg{1,2,3}, tadpole, '
                             'nhanes, synthetic')
    parser.add_argument('--output_path', default='./outputs/%s/',
                        type=str,
                        help='Location where all outputs will be saved.')
    parser.add_argument('--graph_type',
                        default=1,
                        type=int,
                        help='1 for thresholded, 2 for knn, 3 for HEOM')
    # parser.add_argument('--gnn_layers', default=[30, 12, 12, 2], type=int,
    #                     help='input, hidden, and output layer dimensions')
    parser.add_argument('-hp', '--hp_iter', default=100,
                        type=int,
                        help='Number of iterations to run hyperparameter '
                             'search.')
    parser.add_argument('-r', '--remainder', default='passthrough',
                        type=str,
                        help='"drop" or "passthrough". Passthrough will use '
                             'remaining column not specified in column '
                             'transformer. Drop will exclude remaining '
                             'columns.')
    parser.add_argument('--device', default=0, type=int,
                        help='Index of gpu device to use.')
    parser.add_argument('--save_embedding', default=1, type=int,
                        help='Set to 1 to save embedding.')
    parser.add_argument('--p_remain', default=1.0, type=float,
                        help='If 1.0 will not simulate missingess, else if '
                             'between (0,1) will only retain given p_remain '
                             'ration of features and randomly simulate '
                             'missingness.')
    parser.add_argument('--random_search', default=True, type=bool,
                        help='If True, will perform randomized seaerch '
                             'crossvalidation.')
    parser.add_argument('--init_train_labels', default=0, type=int,
                        help='If 1 will initialize training '
                             'labels with actual labels, else replace all '
                             'with zeros.')
    args = parser.parse_args()

    # Parameters used during training for particular session
    args.output_path = os.path.join(args.output_path, 'exp_%03d' % args.exp_num)
    args.cv_type = 2  # 2 for Stratified
    args.cv_folds = 10
    args.rand_seed = 0
    args.baseline_only = True
    args.device = 'cuda:%s' % args.device if torch.cuda.is_available() else \
        'cpu'
    mlp_max_iter = 1000
    args.output_path = args.output_path % args.dataset
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # For reproducibility
    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    random_state = np.random.RandomState(args.rand_seed)
    args.random_state = random_state

    ####################
    # Specify dataset
    ####################
    # TODO refactor to allow others to use their own datasets.
    if args.dataset == 'dizzyreg1':
        dataset = dp.DizzyregDataset(args, 1)
    elif args.dataset == 'dizzyreg2':
        dataset = dp.DizzyregDataset(args, 2)
    elif args.dataset == 'dizzyreg3':
        dataset = dp.DizzyregDataset(args, 3)
    elif args.dataset == 'tadpole':
        dataset = dp.TADPOLEDataset(args)
    elif args.dataset == 'mnist':
        dataset = dp.MNISTDataset(args)
    elif args.dataset == 'nhanes':
        dataset = dp.NHANESDataset(args)
    elif args.dataset == 'synthetic':
        dataset = dp.SyntheticDataset(args)
    else:
        raise NotImplementedError

    #########################
    # Baseline models
    #########################
    cb_earlystop = skorch.callbacks.EarlyStopping(patience=30,
                                                  monitor='valid_loss')
    net = NeuralNetClassifier(
        models.MLPClassifier,
        optimizer=torch.optim.Adam,
        max_epochs=mlp_max_iter,
        device=args.device,
        callbacks=[cb_earlystop],
        lr=0.001,
        module__args=args,
        train_split=CVSplit(5, stratified=True)
        # When data is fitted 20% of which is used as validation
    )
    model_list = [
        net,
        LogisticRegression(),
        RandomForestClassifier(max_depth=5, n_estimators=10),
        KNeighborsClassifier(10),
        SVC(kernel="linear", C=0.025, probability=True),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis()
    ]

    if args.random_search:
        for k, v in enumerate(model_list):
            hyperparam_handler = hparam.HyperParameterProvider(v)
            params = hyperparam_handler.parameters
            model_list[k] = RandomizedSearchCV(v, params,
                                               random_state=args.rand_seed,
                                               n_iter=args.hp_iter)


    trainer = tp.ClassificationTrainer(dataset, model_list, args)
    trainer.train()

    # #########################
    # # Proposed model
    # #########################
    cb_earlystop = skorch.callbacks.EarlyStopping(patience=30,
                                                  monitor='val_acc_crit',
                                                  lower_is_better=False)
    args.len_numerical_features = len(dataset.numerical_idx)
    args.num_class = dataset.data_y.shape[-1]
    args.num_features = dataset.data_x.shape[-1]
    args.num_meta = dataset.adj.shape[-1]

    gnn_net = model_wrapper.TransductiveMGMCModelWrapper(
        models.MGMCCheby,
        optimizer=torch.optim.Adam,
        max_epochs=mlp_max_iter,
        device=args.device,
        callbacks=[('lr_scheduler', LRScheduler(policy=ReduceLROnPlateau)),
                   ('val_acc_crit',
                    skorch.callbacks.EpochScoring(utils.mgmc_val_score,
                                                  name='val_acc_crit',
                                                  use_caching=True,
                                                  on_train=True)),
                   ('stopping_crit', cb_earlystop)
                   ],
        module__args=args,
        batch_size=-1,
        lr=0.001,
        train_split=None,
    )
    # # Transductive model
    model_list = [gnn_net]
    trainer = tp.MGMCTrainer(dataset, model_list, args)
    if args.random_search:
        hyperparam_handler = hparam.HyperParameterProvider(gnn_net)
        params = hyperparam_handler.parameters
        trainer.hp_search(params, algo=hyperopt.rand.suggest,
                          max_evals=args.hp_iter)
    else:
        trainer.train()

    print(time.time() - start_time)

if __name__ == '__main__':
    main()
