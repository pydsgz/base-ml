import abc
import os
import pickle
from glob import glob
import copy
import torch

import numpy as np
import scipy as sp
import pandas as pd
from ray import tune
from hyperopt import hp, Trials, STATUS_OK, STATUS_FAIL, fmin
import hyperopt

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, \
    RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder, \
    FunctionTransformer, StandardScaler, LabelBinarizer
from sklearn.tree import DecisionTreeClassifier

from skorch import NeuralNetRegressor, NeuralNetClassifier, NeuralNet
from skorch.helper import SliceDict

from base_ml import utils
from base_ml import data_provider as dp


class BaseTrainer(abc.ABCMeta):
    # TODO
    @property
    def preprocessor(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self):
        raise NotImplementedError


class MLTrainer:
    """Trainer for Sklearn and PyTorch models.

    Trainer is mainly used for Inductive learning-based models.

    Attributes
        args: Argument parser object.
        model_list: list of sklearn or pytorch models.
        data_provider: base_ml custom data provider object.
        numeric_features: list of indices for numerical features in feature
            matrix.
        categorical_features: list of indices for categorical features in
            feature matrix.
    """
    def __init__(self, data_provider, model_list, args, is_gnn=False):
        super(MLTrainer, self).__init__()
        self.max_trials = None
        self.trials_step = None
        self.args = args
        self.args.is_classification = None
        self.model_list = model_list
        self.data_provider = data_provider
        self.numeric_features = data_provider.numerical_idx
        self.categorical_features = data_provider.non_num_idx
        if data_provider.binary_idx_list is not None:
            self.len_numerical_feat = len(data_provider.numerical_idx) + len(data_provider.binary_idx_list)
        else:
            self.len_numerical_feat = len(data_provider.numerical_idx)
        self.is_gnn = is_gnn
        self.dummy_list = []

    def preprocessor(self, numerical_index=None, categorical_index=None, binary_features=None):
        """Imputation and normalization pre-processing steps.

        Returns
            preprocessor (ColumnTransformer): sklearn transformer to be used
                in a sklearn pipeline.
        """
        trans_list = []
        if numerical_index is not None:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())])
            trans_list.append(('num', numeric_transformer, self.numeric_features))

        if binary_features is not None:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='zzzzz')),
                ('imputer_0.5', FunctionTransformer(utils.take_first_col_and_impute))  # impute missing with 0.5
            ])
            trans_list.append(('binary_feat', categorical_transformer, self.data_provider.binary_idx_list))

        if categorical_index is not None:
            categorical_transformer = Pipeline(steps=[
                ('imputer',
                 SimpleImputer(missing_values=np.nan, strategy='constant',
                               fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            trans_list.append(('cat', categorical_transformer, self.categorical_features))
        preprocessor = ColumnTransformer(
            transformers=trans_list,
            remainder=self.args.remainder,
            sparse_threshold=0)

        return preprocessor

    def train(self):
        cv_folds = self.args.cv_folds
        model_list = self.model_list
        cv_type = self.args.cv_type
        preprocessor = self.preprocessor(numerical_index=self.data_provider.numerical_idx,
                                         categorical_index=self.data_provider.non_num_idx,
                                         binary_features=self.data_provider.binary_idx_list)

        # Cross-validation strategy
        if cv_type == 1:
            cv = KFold(n_splits=cv_folds, random_state=self.args.random_state)
        elif cv_type == 2:
            cv = StratifiedKFold(n_splits=cv_folds, random_state=self.args.random_state, shuffle=True)
        else:
            raise ValueError('Unknown cross-validation strategy.')

        # Train all models in model_list
        for k, clf in enumerate(model_list):
            print('Training %s' % clf)
            data_x = self.data_provider.data_x
            data_y = self.data_provider.data_y

            if self.is_gnn:
                data_adj = self.data_provider.adj.cpu().numpy()

            # Encode non-numerical features before splitting to avoid incorect encoding when splitting dataset within
            # k-fold cross validation procedure.
            nonum_preprocessor = Pipeline([('preprocessing', preprocessor)])
            nonnum_features = nonum_preprocessor.fit_transform(data_x)[:, self.len_numerical_feat:]
            mask_all = self.get_nonnum_observed_mask_matrix(data_x, nonum_preprocessor)

            # Outer-loop cross validation
            y_list = []
            pipe_list = []
            embedding_list = []
            outer_fold_counter = 0
            for train_index, test_index in cv.split(data_x, data_y.argmax(1)):
                print('Running outer fold {}'.format(outer_fold_counter + 1))

                # Initialize pipeline for pre-processing and model.
                steps = list()
                steps.append(("preprocessor", preprocessor))
                as_float = FunctionTransformer(utils.as_float)
                steps.append(('to_float', as_float))
                pipe_preprocess = Pipeline(steps)

                clf_steps = list()
                clf_steps.append(("clf", clf))
                pipe_clf = Pipeline(clf_steps)

                # Split dataset into train and test set
                train_data_x = data_x[train_index, :]
                test_data_x = data_x[test_index, :]
                train_data_y = data_y[train_index]
                test_data_y = data_y[test_index]

                # Pre-process data
                pipe_preprocess.fit(train_data_x)
                train_data_x = pipe_preprocess.transform(train_data_x)
                test_data_x = pipe_preprocess.transform(test_data_x)

                # Use pre-proecssed non-numerical features to avoid incorrect input feature dimension issue.
                nonnum_train_feat = nonnum_features[train_index, :]
                nonnum_test_feat = nonnum_features[test_index, :]
                train_data_x = train_data_x[:, :self.len_numerical_feat]
                test_data_x = test_data_x[:, :self.len_numerical_feat]
                train_data_x = np.concatenate([train_data_x, nonnum_train_feat], 1).astype(np.float32)
                test_data_x = np.concatenate([test_data_x, nonnum_test_feat], 1).astype(np.float32)

                # Which entries are observed and imputed
                train_mask = mask_all[train_index, :]
                test_mask = mask_all[test_index, :]

                if self.is_gnn:
                    train_adj = data_adj[train_index, :][:, train_index]
                    train_dataset = dp.InductiveGNNDataset(train_data_x, train_adj, train_data_y, train_mask)
                    train_data_x = train_dataset

                    test_adj = data_adj[test_index, :][:, test_index]
                    test_dataset = dp.InductiveGNNDataset(test_data_x, test_adj, test_data_y, test_mask)
                    test_data_x = test_dataset

                # Train model
                print('Fitting model parameters...')
                pipe_clf.fit(train_data_x, train_data_y.argmax(-1))

                # Predict test set
                print('Evaluating model on test set...')
                cur_pred_y = pipe_clf.predict(test_data_x)
                if self.args.is_classification:
                    cur_pred_y_proba = pipe_clf.predict_proba(test_data_x)
                else:
                    cur_pred_y_proba = None
                y_true_y_hat = (test_data_y.argmax(-1), cur_pred_y_proba, cur_pred_y)
                y_list.append(y_true_y_hat)
                pipe_list.append(copy.deepcopy(pipe_clf))
                outer_fold_counter += 1

                if self.args.save_embedding and isinstance(clf, NeuralNetClassifier):
                    cur_state_dict = pipe_clf.named_steps['clf'].module_.enc.state_dict().copy()
                    embedder = NeuralNetClassifier(pipe_clf.named_steps['clf'].module_.enc)
                    embedder.initialize()
                    embedder.module_.load_state_dict(cur_state_dict)
                    test_set_embedding = embedder.forward(test_data_x)
                    test_set_embedding = test_set_embedding.numpy()
                    embedding_list.append(test_set_embedding)

            # Save all test prediction results
            estimator_name = get_estimator_name(clf)

            # Save output predictions
            output_pickle_path = os.path.join(self.args.output_path,
                                              "output_pred_%s_.pkl" % (
                                                  estimator_name))
            output_pickle_path.replace('.pkl', 'fold')
            with open(output_pickle_path, 'wb') as pfile:
                pickle.dump(y_list, pfile, protocol=pickle.HIGHEST_PROTOCOL)

            # Save model
            save_model_path = os.path.join(self.args.output_path,
                                           "model_%s.pkl" % estimator_name)
            with open(save_model_path, 'wb') as pfile:
                pickle.dump(pipe_list, pfile, protocol=pickle.HIGHEST_PROTOCOL)

            # Save pre-output layer embedding
            if self.args.save_embedding and isinstance(clf, NeuralNetClassifier):
                print('Saving embedding...')
                np.save(save_model_path.replace('.pkl', 'embedding.npy'), np.concatenate(embedding_list), 0)
                if os.path.exists(save_model_path.replace('.pkl', 'embedding.npy')):
                    print('Embedding saved...')



    def create_output_folder(self):
        if not os.path.exists(self.args.output_path):
            os.makedirs(self.args.output_path)

    def get_nonnum_observed_mask_matrix(self, feat_data, preprocessor_pipe):
        """Get the matrix specifying which are observed entries."""

        offset_idx = len(self.data_provider.numerical_idx)
        is_nan_mat = pd.DataFrame(feat_data).isna().to_numpy()
        col_idx = 0 + offset_idx

        # Check if there are non-numerical features
        nonnum_preprocessor = preprocessor_pipe.steps[0][1].named_transformers_
        if 'cat' in nonnum_preprocessor:
            categories_ = nonnum_preprocessor['cat'].named_steps['onehot'].categories_
            # Expand column based on how many additional columns were after pre_processing non-numerical columns.
            mask_list = []
            for k, v in enumerate(self.data_provider.non_num_idx):
                cur_arr = is_nan_mat[:, v]
                num_columns = len(categories_[k])
                cur_mask = np.tile(cur_arr.reshape(-1, 1), (1, num_columns))
                mask_list.append(cur_mask)

            non_num_mask_matrix = np.concatenate(mask_list, 1)
            if self.data_provider.binary_idx_list is not None:
                numerical_idx_ = np.concatenate([self.data_provider.numerical_idx, self.data_provider.binary_idx_list])
            else:
                numerical_idx_ = self.data_provider.numerical_idx
            num_mask_matrix = is_nan_mat[:, numerical_idx_]
            mask_matrix = np.concatenate([num_mask_matrix, non_num_mask_matrix], 1)
        else:
            mask_matrix = is_nan_mat
        return ~mask_matrix


class ClassificationTrainer(MLTrainer):
    def __init__(self, data_provider, model_list, args, is_gnn=None):
        super(ClassificationTrainer, self).__init__(data_provider, model_list, args, is_gnn=is_gnn)
        self.args.is_classification = True
        self.create_output_folder()


class TransductiveGNNClassificationTrainer(MLTrainer):
    def __init__(self, **kwargs):
        super(TransductiveGNNClassificationTrainer, self).__init__(**kwargs)
        self.args.is_classification = True
        self.create_output_folder()
        self.args = kwargs.get('args')

    def train(self):
        """Transductive classification training workflow."""
        # random_state = np.random.RandomState(self.args.rand_seed)
        cv_folds = self.args.cv_folds
        model_list = self.model_list
        cv_type = self.args.cv_type
        preprocessor = self.preprocessor(numerical_index=self.data_provider.numerical_idx,
                                         categorical_index=self.data_provider.non_num_idx,
                                         binary_features=self.data_provider.binary_idx_list)

        # Cross-validation strategy
        if cv_type == 1:
            cv = KFold(n_splits=cv_folds, random_state=self.args.random_state)
        elif cv_type == 2:
            cv = StratifiedKFold(n_splits=cv_folds, random_state=self.args.random_state)
        else:
            raise ValueError('Unknown cross-validation strategy.')

        # Train all models in model_list
        for k, clf in enumerate(model_list):
            # List of steps to plug into pipeline
            print('Training %s' % clf)
            steps = list()
            steps.append(("preprocessor", preprocessor))
            as_float = FunctionTransformer(utils.as_float)
            steps.append(('to_float', as_float))
            preprocessor_pipe = Pipeline(steps)

            pipe = Pipeline([("clf", clf)])

            data_x = self.data_provider.data_x
            data_y = self.data_provider.data_y
            data_adj = self.data_provider.adj.cpu().numpy()

            # Outer-loop cross validation
            y_list = []
            embedding_list = []
            train_cv = StratifiedKFold(n_splits=5,
                                       random_state=self.args.random_state)
            outer_fold_counter = 0
            for train_index, test_index in cv.split(data_x, data_y.argmax(1)):
                print('Running outer fold {}'.format(outer_fold_counter + 1))

                tr_split = train_cv.split(train_index, data_y[train_index].argmax(1))
                tr_idx, val_idx = list(tr_split)[0]

                val_index = train_index[val_idx]
                train_index = train_index[tr_idx]

                train_x = data_x
                train_y = data_y
                test_x = data_x
                test_y = data_y

                preprocessor_pipe.fit(train_x)
                train_x = preprocessor_pipe.transform(train_x)
                test_x = preprocessor_pipe.transform(test_x)

                # Which entries are observed and imputed
                mask_matrix = self.get_nonnum_observed_mask_matrix(data_x, preprocessor_pipe)

                train_adj = data_adj
                test_adj = data_adj
                train_dataset = dp.TransductiveGNNDataset(train_x, train_adj, train_y,
                                                          train_index, val_index, mask_matrix)
                test_dataset = dp.TransductiveGNNDataset(test_x, test_adj, test_y,
                                                         test_index, val_index, mask_matrix)

                # Estimate parameters
                pipe.fit(train_dataset, train_y)

                # Test set preddictions
                if isinstance(pipe.named_steps['clf'], (RandomizedSearchCV, GridSearchCV)):
                    cur_pred_y = pipe.named_steps['clf'].best_estimator_.forward(test_dataset)
                else:
                    cur_pred_y = pipe.named_steps['clf'].forward(test_dataset)
                test_pred_prob_target = cur_pred_y[test_index]
                test_pred_target = test_pred_prob_target.argmax(-1)
                test_gt_target = test_y[test_index].argmax(-1)

                print('===' * 10)
                print('Test-set Accuracy = %s' % accuracy_score(test_gt_target, test_pred_target))
                print('Test-set Confusion Matrix')
                print(confusion_matrix(test_gt_target, test_pred_target))
                print('===' * 10)

                y_true_y_hat = (test_gt_target,
                                test_pred_prob_target,
                                test_pred_target)
                y_list.append(y_true_y_hat)

            # Save output predictions
            estimator_name = get_estimator_name(clf)
            output_pickle_path = os.path.join(self.args.output_path,
                                              "output_pred_%s_.pkl" % (
                                                  estimator_name))
            with open(output_pickle_path, 'wb') as pfile:
                pickle.dump(y_list, pfile, protocol=pickle.HIGHEST_PROTOCOL)

            # Save model
            save_model_path = os.path.join(self.args.output_path,
                                           "model_%s.pkl" % estimator_name)
            with open(save_model_path, 'wb') as pfile:
                pickle.dump(pipe, pfile, protocol=pickle.HIGHEST_PROTOCOL)


class MGMCTrainer(MLTrainer):
    def __init__(self, data_provider, model_list, args):
        super(MGMCTrainer, self).__init__(data_provider, model_list, args)
        self.args.is_classification = True
        self.create_output_folder()
        self.args = args
        self.average_val_acc = None

    def train(self):
        """Transductive training workflow."""

        # self.args.ce_wt = config["ce_wt"]
        # self.args.frob_wt = config["frob_wt"]
        # self.args.dirch_wt = config["dirch_wt"]

        # random_state = np.random.RandomState(self.args.rand_seed)
        cv_folds = self.args.cv_folds
        model_list = self.model_list
        cv_type = self.args.cv_type
        preprocessor = self.preprocessor(numerical_index=self.data_provider.numerical_idx,
                                         categorical_index=self.data_provider.non_num_idx,
                                         binary_features=self.data_provider.binary_idx_list)

        # Cross-validation strategy
        if cv_type == 1:
            cv = KFold(n_splits=cv_folds, random_state=self.args.random_state)
        elif cv_type == 2:
            cv = StratifiedKFold(n_splits=cv_folds, random_state=self.args.random_state)
        else:
            raise ValueError('Unknown cross-validation strategy.')

        # Train all models in model_list
        for k, clf in enumerate(model_list):
            # List of steps to plug into pipeline
            print('Training %s' % clf)
            steps = list()
            steps.append(("preprocessor", preprocessor))
            as_float = FunctionTransformer(utils.as_float)
            steps.append(('to_float', as_float))
            preprocessor_pipe = Pipeline(steps)

            pipe = Pipeline([("clf", clf)])

            data_x = self.data_provider.data_x
            data_y = self.data_provider.data_y
            data_adj = self.data_provider.adj.cpu().numpy()

            # Outer-loop cross validation
            y_list = []
            pipe_list = []
            val_acc_list = []
            train_cv = StratifiedKFold(n_splits=5,
                                       random_state=self.args.random_state)

            for train_index, test_index in cv.split(data_x, data_y.argmax(1)):

                tr_split = train_cv.split(train_index, data_y[train_index].argmax(1))
                tr_idx, val_idx = list(tr_split)[0]

                val_index = train_index[val_idx]
                train_index = train_index[tr_idx]

                train_x = data_x
                train_y = data_y
                test_x = data_x
                test_y = data_y

                preprocessor_pipe.fit(train_x)
                train_x = preprocessor_pipe.transform(train_x)
                test_x = preprocessor_pipe.transform(test_x)

                # Which entries are observed and imputed
                mask_matrix = self.get_nonnum_observed_mask_matrix(data_x, preprocessor_pipe)

                train_adj = data_adj
                test_adj = data_adj
                train_dataset = dp.TransductiveMGMCDataset(train_x, train_adj, train_y,
                                                           train_index, val_index, mask_matrix, is_training=True,
                                                           args=self.args)
                test_dataset = dp.TransductiveMGMCDataset(test_x, test_adj, test_y,
                                                          test_index, val_index, mask_matrix, is_training=False,
                                                          args=self.args)

                # Estimate parameters
                pipe.fit(train_dataset, train_y)

                # Store validation set metrics for hyperparameter search
                val_dataset = dp.TransductiveMGMCDataset(test_x, test_adj, test_y,
                                                         val_index, val_index, mask_matrix, is_training=False,
                                                         args=self.args)
                val_pred = pipe.named_steps['clf'].forward(val_dataset)
                num_class = test_y.shape[-1]
                val_gt_target = test_y[val_index].argmax(-1)
                val_pred_prob_target = torch.nn.functional.softmax(val_pred[0][val_index, -num_class:])
                val_pred_target = val_pred_prob_target.argmax(-1)
                val_acc_list.append(accuracy_score(val_gt_target, val_pred_target))

                # Test set preddictions
                if isinstance(pipe.named_steps['clf'], (RandomizedSearchCV, GridSearchCV)):
                    cur_pred_y = pipe.named_steps['clf'].best_estimator_.forward(test_dataset)
                else:
                    cur_pred_y = pipe.named_steps['clf'].forward(test_dataset)
                num_class = test_y.shape[-1]
                test_pred_prob_target = torch.nn.functional.softmax(cur_pred_y[0][test_index, -num_class:])
                test_pred_target = test_pred_prob_target.argmax(-1)
                test_gt_target = test_y[test_index].argmax(-1)

                print('===' * 10)
                print('Test-set Accuracy = %s' % accuracy_score(test_gt_target, test_pred_target))
                print('Test-set Confusion Matrix')
                print(confusion_matrix(test_gt_target, test_pred_target))
                print('===' * 10)

                # y_true_y_hat = (test_y[test_index].argmax(1),
                #                 test_pred_prob_target,
                #                 test_pred_target)

                if self.args.p_remain < 1.0:
                    mask = np.zeros_like(test_x)
                    mask[test_dataset.missing_idx_sim] = 1
                    pred_imputation = cur_pred_y[0][:, :-num_class].numpy()
                    gt_imputation = test_x
                    diff = ((pred_imputation - gt_imputation)*mask)[test_index, :]
                    # diff = (pred_imputation - gt_imputation)[test_index, :]
                    test_rmse = np.sqrt(np.mean(diff**2))
                    # test_rmse = np.sqrt((diff**2).sum()/mask[test_index, :].sum())
                    y_true_y_hat = (test_y[test_index].argmax(1),
                                    test_pred_prob_target,
                                    test_pred_target, test_rmse)
                else:
                    y_true_y_hat = (test_y[test_index].argmax(1),
                                    test_pred_prob_target,
                                    test_pred_target, None)

                y_list.append(y_true_y_hat)
                pipe_list.append(copy.deepcopy(pipe))

            # Report average validation set same as MGMC paper for now for comparability
            self.average_val_acc = -np.mean(val_acc_list)

            # Save output predictions
            estimator_name = get_estimator_name(clf)
            output_pickle_path = os.path.join(self.args.output_path,
                                              "output_pred_%s_%sremain.pkl" % (estimator_name,
                                                                               str(self.args.p_remain*100)))
            with open(output_pickle_path, 'wb') as pfile:
                pickle.dump(y_list, pfile, protocol=pickle.HIGHEST_PROTOCOL)

            # Save model
            save_model_path = os.path.join(self.args.output_path,
                                           "model_%s_%sremain.pkl" % (estimator_name, str(self.args.p_remain * 100)))
            with open(save_model_path, 'wb') as pfile:
                pickle.dump(pipe_list, pfile, protocol=pickle.HIGHEST_PROTOCOL)


    def hp_search(self, parameters_space):
        """Search hyperparamters given model and hyperparater space."""
        print('Running hyperparam search...')
        # how many additional trials to do after loading saved
        trials_step = self.trials_step

        # Use something small to not have to wait
        max_t = self.max_trials
        pickle_str = f"{self.model_list[0].module.__name__}_{self.args.dataset}"
        self.pickle_str = f"{self.args.output_path}/{pickle_str}_hperopt_out_trials.pkl"


        assert len(self.model_list) == 1, "Currently only supports single models training and hp-search."
        # Perform HP search
        if self.args.p_remain == 1.0 and not os.path.exists(self.pickle_str):
            try:
                with open(self.pickle_str, "wb") as f:
                    trials = pickle.load(f)
                print("Found saved Trials! Loading...")
                print("Rerunning from {} trials".format(len(trials.trials)))
            except Exception as err:
                print(err)
                trials = Trials()
                print("Starting from scratch: new trials created.")

            best_model = hyperopt.fmin(self.hp_objective, parameters_space, trials=trials,
                                       algo=hyperopt.tpe.suggest,
                                       max_evals=120)
            with open(self.pickle_str, "wb") as f:
                pickle.dump(trials, f)
            print("HP search done...")
        else:
            # Rerun training using best hyperparameters
            with open(self.pickle_str, "rb") as f:
                best_hyperparams = pickle.load(f)
                best_hyperparams = best_hyperparams.best_trial['result']['space']
                self.train_with_param(best_hyperparams)

    def hp_objective(self, params):
        try:
            train_res = self.eval_with_param(params)
            print("Mean bce {}".format(train_res))
            return {'loss': train_res, 'status': STATUS_OK, 'space': params}
        except Exception as err:
            print(err)
            return {'loss': np.nan, 'status': STATUS_FAIL, 'space': params}

    def train_with_param(self, params):
        self.args.ce_wt = params["ce_wt"]
        self.args.frob_wt = params["frob_wt"]
        self.args.dirch_wt = params["dirch_wt"]
        self.train()
        print("Training done!")

    def eval_with_param(self, params):
        """Evaluate given current parameters on train and validation set."""
        self.args.ce_wt = params["ce_wt"]
        self.args.frob_wt = params["frob_wt"]
        self.args.dirch_wt = params["dirch_wt"]
        print(f"Training...")
        self.train()
        return self.average_val_acc


def get_estimator_name(estimator_object):
    """Get estimator name given pipe object or estimator object from Sklearn

    :param estimator_object:
    :return:
    """
    if isinstance(estimator_object, (RandomizedSearchCV, GridSearchCV)):
        if isinstance(estimator_object, NeuralNet):
            estimator_name = estimator_object.estimator.module.__name__
        else:
            estimator_name = estimator_object.estimator.__class__.__name__

    elif isinstance(estimator_object, NeuralNet):
        estimator_name = estimator_object.module_.__class__.__name__

    elif isinstance(estimator_object, (RandomForestClassifier, LogisticRegression, KNeighborsClassifier, SVC,
                                       GaussianProcessClassifier, DecisionTreeClassifier, AdaBoostClassifier,
                                       GaussianNB, QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis)):
        estimator_name = estimator_object.__class__.__name__
    else:
        raise NotImplementedError('Estimator name not yet implemented.')
    return estimator_name
