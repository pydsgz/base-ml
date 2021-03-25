import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd

import torch
# from statannot import add_stat_annotation
from scipy.stats import shapiro, ttest_rel, ttest_ind, mannwhitneyu, wilcoxon, f_oneway, friedmanchisquare, kruskal
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from skorch import callbacks
import skorch
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, \
    roc_auc_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
# from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler, LabelBinarizer, OneHotEncoder, FunctionTransformer
import umap

import data_provider as dp

try:
    import seaborn as sns
    # from rpy2.robjects.packages import importr
    # from rpy2.robjects.numpy2ri import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    import pingouin as pg

    stats = importr('stats')
except Exception as e:
    print(e)

try:
    from captum.attr import IntegratedGradients
except Exception as e:
    print(e)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          filename=None):
    """This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    # TODO
    Args:
    Returns:

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if normalize is True:
        vmin = 0.0
        vmax = 1.0
    else:
        vmin = None
        vmax = None

    classes = [x[2:] if ('0_' in x or '1_' in x) else x for x in \
               classes]

    hFig = plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.tight_layout()
    plt.show()
    if filename is not None:
        # hFig.set_size_inches(8,8)
        # plt.tight_layout()
        hFig.savefig(filename, dpi=300)
    return hFig


class MLOutputHandler:
    def __init__(self):
        pass

    @staticmethod
    def to_dataframe(y_list, estimator_name):
        # Calculate metrics
        rmse = [np.sqrt(mean_squared_error(x[0], x[2])) for x in
                y_list]
        cor_res = [stats.pearsonr(x[0].squeeze(), x[2].squeeze()) for x in
                   y_list]
        ccoef = [x[0] for x in cor_res]
        p_val = [x[1] for x in cor_res]
        r2_ = [r2_score(x[0], x[2]) for x in y_list]

        # Combine results as dataframe
        rmse = pd.DataFrame(rmse)
        corel = pd.DataFrame(ccoef)
        r2_ = pd.DataFrame(r2_)
        pval_ = pd.DataFrame(p_val)
        new_df = pd.concat([rmse, r2_, corel, pval_], 1)
        new_df.columns = ['RMSE', 'R2_score', 'Correlation_Coefficient',
                          'p_val']
        new_df = new_df.round(5)

        # Prettify dataframe for easy plotting
        new_df.reset_index(inplace=True)
        cols_to_melt = new_df.columns
        new_df = new_df.melt('index', list(cols_to_melt[1:]))
        new_df['estimator'] = estimator_name
        return new_df


class ClassifierOutputHandler(MLOutputHandler):
    def __init__(self):
        super(ClassifierOutputHandler, self).__init__()

    @staticmethod
    def to_dataframe(y_list, estimator_name, average=None):
        """Given a list of output results calculate metrics and return as
        dataframe."""
        n_class = y_list[0][1].shape[-1]
        multi_class = None if n_class == 2 else 'ovr'

        # Calculate metrics
        acc = [accuracy_score(x[0], x[2]) for x in y_list]
        # mat_coef = [matthews_corrcoef(x[0], x[2]) for x in y_list]
        if n_class == 2:
            fscore = [f1_score(x[0], x[2], average=average) for x in
                      y_list]
            auc_ = [roc_auc_score(x[0], x[1][:, 1:], average=average,
                                  multi_class=multi_class)
                    for x in y_list]
        else:
            fscore = [f1_score(x[0], x[2], average=average) for x in y_list]
            auc_ = [roc_auc_score(x[0], x[1], average=average,
                                  multi_class=multi_class) for x in
                    y_list]
        acc = pd.DataFrame(acc)
        fscore = pd.DataFrame(fscore)
        auc_ = pd.DataFrame(auc_)
        # mat_coef = pd.DataFrame(mat_coef)

        # Combine all dataframe
        new_df = pd.concat([acc, fscore, auc_], 1)
        new_df.columns = ['Accuracy', 'F1_score', 'ROC_AUC']
        new_df = new_df.round(5)

        # Prettify dataframe for easy plotting
        new_df.reset_index(inplace=True)
        cols_to_melt = new_df.columns
        new_df = new_df.melt('index', list(cols_to_melt[1:]))
        new_df['estimator'] = estimator_name
        return new_df

    @staticmethod
    def pretty_df_to_boxplots(res_df, order=None, color_pal=None, y_limit=None):
        """Given wide dataframe convert to boxplots."""
        sns.reset_defaults()
        sns.set_style("whitegrid")
        sns.set(font_scale=2.5)
        plt.figure(figsize=(32, 8))
        plt.axis('tight')

        metric_names = res_df.variable.unique()
        len_ = len(metric_names)
        list_of_fig = []
        for k, cur_variable in enumerate(metric_names):
            cur_data = res_df[res_df.variable == cur_variable]
            if cur_variable == 'Accuracy':
                cur_data.value *= 1.
                # cur_y_limit = (y_limit[0] * 100.0, y_limit[1] * 100.0)
                # cur_data.value * 1.
            elif cur_variable == 'F1_score':
                # print(cur_data.value)
                cur_data.value = cur_data.value.round(2)
                # print(cur_data.value)
            plt.subplot(1, len_, k + 1)
            ax = sns.boxplot(x="variable", y="value",
                             hue="estimator",
                             data=cur_data,
                             palette='colorblind')
            ax.set(xticklabels=[])
            ax.legend(title='Models')
            ax.set_ylim(y_limit[0], y_limit[1])
            if k == len(metric_names) - 1:
                ax.legend(bbox_to_anchor=(1.1, 1.05))
            else:
                cur_legend_ = ax.legend()
                cur_legend_.remove()
            # ax.legend(loc='upper left', fancybox=True, ncol=3,
            #           bbox_to_anchor=(.01, y_limit[1]+.20))

            # add_stat_annotation(ax, data=cur_data,
            #                     x="variable", y="value", hue="estimator",
            #                     # box_pairs=[((cur_variable, "5-ANN"),
            #                     #             (cur_variable, "LR"))],
            #                     box_pairs=[(("LR", "5-ANN"))],
            #                     test='Wilcoxon', text_format='star',
            #                     loc='outside', verbose=2)


            # else:
            #     cur_legend_ = ax.legend()
            #     cur_legend_.remove()
            plt.xlabel('')
            plt.ylabel(cur_variable)
            plotter = sns.categorical._BoxPlotter(x="variable", y="value",
                                                  hue="estimator",
                                                  data=res_df.copy(),
                                                  order=None,
                                                  hue_order=order,
                                                  palette=color_pal,
                                                  orient=None,
                                                  width=.8, color=None,
                                                  saturation=.75, dodge=True,
                                                  fliersize=5, linewidth=None)
            list_of_fig.append((plotter, ax))
            # list_of_fig.append(ax)
        plt.show()
        return list_of_fig

    @staticmethod
    def to_confusion_matrix(y_list, col_name, class_names=None, normalize=True):
        # Plot confusion matrix
        sns.reset_defaults()
        plt.figure(figsize=(20, 6))
        cur_true_labels = [x[0] for x in y_list]
        cur_true_labels = np.concatenate(cur_true_labels)
        cur_pred_labels = [x[2] for x in y_list]
        cur_pred_labels = np.concatenate(cur_pred_labels)
        cm = confusion_matrix(cur_true_labels, cur_pred_labels)
        # if len(np.unique(cur_true_labels)) == 2:
        cm_fig = plot_confusion_matrix(cm,
                                       classes=class_names, title=col_name,
                                       normalize=normalize)
        # elif len(np.unique(cur_true_labels)) == 3:
        #     cm_fig = plot_confusion_matrix(cm,
        #                                    classes=class_names, title=col_name,
        #                                    normalize=normalize)
        # else:
        #     raise NotImplementedError
        # cm_fig.savefig(output_pickle_path.replace('.pkl', '_cm.png'))

        return cm_fig


def get_k_nearest_neighbors(X, K, metric=None):
    """
    Get K nearest neighbors.

    Parameters:
    X: np.ndarray
        Numpy array with n by d dimension where n
        is the number of samples/nodes and d is the
        number of features
    K: int
        Number of nearest neighbors to find.
    metric

    Returns:
    res_X: np.ndarray
        Indeces of numpy array with n by K dimension. Where n are the
        number of samples and K are the number of neighbors.
    """
    K = K + 1
    if metric is None:
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree',
                                n_jobs=5).fit(X)
    else:
        nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree',
                                metric=metric, n_jobs=5).fit(X)
    indices = nbrs.kneighbors(X, return_distance=False)
    return indices[:,
           1:]  # exclude the first column as these are just row indices


def get_adjacency_matrix(X, indices):
    """
    Get adjacency matrix of X with size n by k (k number of
    connections/neighbors). Output matrix will be a square
    matrix of n by n where n is the number of nodes.

    Parameter:
    X: numpy.ndarray
        n by k matrix
    indices: numpy.ndarray

    Returns:
    res_A: np.matrix
        n by n numpy
    """
    res_A = np.zeros((X.shape[0], X.shape[0]))
    for k, v in enumerate(indices):
        np.put(res_A[k], v, 1)
    return np.matrix(res_A)


def as_float(x):
    return x.astype(np.float32)


class InputShapeSetter(skorch.callbacks.Callback):
    # Dynamically set input-layer dimension using a callback on train begin.
    def on_train_begin(self, net, X, y):
        net.set_params(module__input_dim=X.shape[1])


class OutputShapeSetter(skorch.callbacks.Callback):
    # Dynamically set output-layer dimension using a callback on train begin.
    def on_train_begin(self, net, X, y):
        net.set_params(module__output_dim=y.shape[1])


def plot_imp_features_svm(classifier, feature_names, top_features=20,
                          classes=1):
    """
    Plot feature importance of SVM linear classifier.

    Parameters:
        classifier: sklearn classifier
            SVM classifier
        feature_names: list of str
            List of features names in dataframe.
        top_features: int
            Number of importance features to show.

    Returns:
        res_list: list of str
            List of strings which are important.
    """
    feat_list = []
    for k, v in enumerate(range(classes)):
        print('Feature coef for class %s' % k)
        coef = classifier.coef_[k:k + 1, :].ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack(
            [top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients],
                color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features),
                   feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()
        res_list = list(feature_names[top_coefficients])
        feat_list.append(res_list)
    return feat_list


def get_feat_imp(classifier, feature_names, top_features=20,
                 classes=1):
    feat_list = []
    for k, v in enumerate(range(classes)):
        print('Feature coef for class %s' % k)
        coef = classifier.coef_[k:k + 1, :].ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack(
            [top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients],
                color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features),
                   feature_names[top_coefficients], rotation=60, ha='right')
        plt.show()
        res_list = list(feature_names[top_coefficients])
        feat_list.append(res_list)
    return feat_list


def plot_imp_features_rf(classifier, col_names, top_features=10):
    """
    Plot feature importance of a random forest classifier.

    Parameters:
    classifier: sklearn classififer
        Random forest classifier
    feature_names: list of str
        List of feature names
    top_features: int
        Number of features

    Returns:
        col_names: list of str
            List of features with highest importance.
    """
    importances = classifier.feature_importances_
    indices = np.argsort(importances)[::-1]
    indices = indices[:top_features]
    importances = importances[indices]
    col_names = np.array(col_names)
    col_names = list(col_names[indices])
    # col_names = [x.upper() for x in col_names]
    y_pos = np.arange(len(col_names))
    fig, ax = plt.subplots()
    ax.barh(y_pos, importances, align='center',
            color='gray', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(col_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance')
    plt.show()
    return list(col_names), importances


def barplot_feature_attr(attr_arr, column_names, top_k=10):
    """Plot top-k feature attributions.

    Args:
        attr_arr (list): Array of feature attribution.
        column_names (list): Array of actual features' names.
        top_k (int): Number of top-k feature attributes to plot.

    Returns:
    """
    top_features = 10
    indices = np.argsort(attr_arr)[::-1]
    indices = indices[:top_k]
    attr_arr = attr_arr[indices]
    col_names = np.array(column_names)
    col_names = list(col_names[indices])
    # col_names = [x.upper() for x in col_names]
    y_pos = np.arange(len(col_names))
    fig, ax = plt.subplots()
    ax.barh(y_pos, attr_arr, align='center',
            color='gray', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(col_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    # ax.set_xlabel('Feature Importance')
    ax.set_title('Mean feature attribution')
    plt.show()


def annotate_all_figures(best_model, data_df, fig_list, model_names,
                         y_axis=1.01, rmse_mode=False, iter_list=None,
                         metric_list=None, y_limit=None):
    """
    Annotate all given figures based on p-val <= 0.05. Will add an asterisk
    above the box when there is a significant difference between comparison
    model vs best model.

    Parameters
    ----------
    best_model: str
        Model name of best model
    data_df: pandas.DataFrame
        Dataframe containing all metric results.
    fig_list: list of tuple (_Boxplotter, plt.Figure)
        Output from pretty_box_plots.
    rmse_mode: bool
        If True, will annotate RMSE figure only.

    Returns
    -------
    new_fig_list: list of plt.Figure
        Annotated list of figures.
    """
    # if rmse_mode:
    #     iter_list = [25, 50, 75]
    # else:
    #     iter_list = [25, 50, 75, 100]

    all_p_val_df = get_all_p_values(model_names, best_model, data_df,
                                    iter_list, metric_list)
    print(all_p_val_df)
    bool_all_pval_df = all_p_val_df <= 0.05
    new_fig_list = []
    # rmse_y_min= data_df.RMSE.std()
    # Annotate each figuannotate_all_figuresre excluding
    # figure with RMSE as title/metric_name.
    for i in fig_list:
        plt.clf()
        cur_bp = i[0]
        cur_fig = i[1]
        # print(len(cur_fig.get_axes()))
        # cur_ax = cur_fig.get_axes()[0]
        # metric_name = cur_ax.get_title()
        print(cur_fig.get_ylabel())
        metric_name = cur_fig.get_ylabel()

        if rmse_mode is False and metric_name == 'RMSE':
            pass
        else:
            is_sig_dict = get_model_names(bool_all_pval_df, metric_name,
                                          iter_list)
            # import pdb; pdb.set_trace()
            if metric_name == 'Accuracy':
                # new_fig = plot_asterisk(cur_bp, cur_fig, is_sig_dict,
                #                         y_axis=y_axis + 100)
                new_fig = plot_asterisk(cur_bp, cur_fig, is_sig_dict,
                                        y_axis=y_axis)
                new_fig_list.append(new_fig)
            else:
                new_fig = plot_asterisk(cur_bp, cur_fig, is_sig_dict,
                                        y_axis=y_axis)
                new_fig_list.append(new_fig)

    return new_fig_list


def get_model_names(bool_df, metric_name, p_list=[25, 50, 75, 100]):
    """
    Given dataframe of booleans, return the column name for every index where
    value is True. {25:['Mean+LR'], 50:['Mean+LR'], 100:['Mean+LR',
    'MICE+RF']}

    Parameters
    ----------
    bool_df: pandas.DataFrame
        Dataframe containing booleans. Where major index in
        classification metric name, minor index is percentage of available
        entries, and column names are model names.
    metric_name: str
        Major index name of interest.
    p_list: list of ints
        [25,50,75,100]

    Returns
    -------
    cur_dict: dict
        {25:['Mean+LR'], 50:['Mean+LR'], 100:['Mean+LR', 'MICE+RF']}
    """
    cur_dict = dict()
    for i in p_list:
        cur_df = bool_df.loc[metric_name]
        col_names = cur_df.columns
        # print(metric_name, i)
        mask = list(cur_df.loc[i])
        cur_model_names = list(col_names[mask])
        cur_dict[i] = cur_model_names
    return cur_dict


def plot_asterisk(cur_bp, cur_fig, is_sig_dict, y_axis=1.03, str_text='*'):
    """
    Plot asterisk above the axis given dict of model names.

    Parameters
    ----------
    cur_bp: seaborn.categorical._Boxplotter
        Output from seaborn _Boxplotter class. Will use to get x-positions.
    cur_fig: matplotlib.figure.Figure
        Figure to annotate on. Output from seaborn plotter.
    is_sig_dict: dict
        {25: ['model_name1', 'model_name2'], 50:['model_name2']}
    y_axis: float
        Location in y dimension.
    str_text: str
        Text to annotate on figure.

    Returns
    -------
    cur_fig: matplotlib.figure.Figure
        Returns the annotated figure.
    """
    # cur_ax = cur_fig.get_axes()[0]
    cur_ax = cur_fig.axes

    # For every percentage
    for i in is_sig_dict:
        cur_val = is_sig_dict[i]

        # For every model name at current percentage annotate asterisk
        for j in cur_val:
            # x_pos = cur_bp.group_names.index(i) + cur_bp.hue_offsets[
            #     cur_bp.hue_names.index(j)]
            x_pos = cur_bp.hue_offsets[cur_bp.hue_names.index(j)]
            cur_ax.text(x_pos, y_axis, str_text)
    return cur_fig


def get_p_value(df, best_model, compare_model, list_iter, metric_list=None):
    """
    Check if best_model is significantly different with compare_model. Will
    compute p-values for all classification metrics.

    Parameters
    ----------
    df: pd.DataFrame
    best_model: str
        Best model's name in model column of df.
    compare_model: str
        Compare with model's name in model column of df.
    Returns
    -------
    res_: dict
        {'model_name': {'metric1':p-value, 'metric2':p-value}}
    """
    stat_metric = stats.wilcox_test
    res_dict = {}
    for i in list_iter:
        cur_dict = {}
        print(best_model, 'vs', compare_model)
        df_1 = df[(df['%'] == i) & (df.model == best_model)]
        df_2 = df[(df['%'] == i) & (df.model == compare_model)]
        for j in metric_list:
            cur_d1 = np.array(df_1.value[df_1.variable == j])
            cur_d2 = np.array(df_2.value[df_2.variable == j])
            if j == 'RMSE':
                cur_d1 = np.nan_to_num(cur_d1)
                cur_d2 = np.nan_to_num(cur_d2)
            print(i, j)
            print(cur_d1)
            print(cur_d2)

            numpy2ri.activate()
            # cur_d1 = numpy2ri(cur_d1.copy())
            # cur_d2 = numpy2ri(cur_d2.copy())
            cur_d1 = cur_d1.copy()
            cur_d2 = cur_d2.copy()
            res_p = stat_metric(cur_d1, cur_d2, paired=True)
            res_p = np.array(res_p[2])[0]
            numpy2ri.deactivate()
            cur_dict[j] = res_p
        res_dict[i] = cur_dict
    res_ = dict()
    res_[compare_model] = res_dict
    return res_


def get_all_p_values(list_model_names, best_model_name, dframe, list_iter,
                     metric_list=None):
    """
    Compare if given list of models is significantly different with the
    best_model_name. Default p-value threshold is 0.05.

    Parameters
    ----------
    list_model_names: list of str
        List of model names.
    best_model_name: str
        Best model name to compare against.
    dframe: pandas.DataFrame
        Dataframe containing all information.
    Returns
    -------
    new_df: pandas.DataFrame
        Dataframe containing all p-values.
    """
    p_val_all = {}
    for i in list_model_names:
        cur_val_ = get_p_value(dframe, best_model_name, i, list_iter,
                               metric_list)
        p_val_all.update(cur_val_)
    new_df = pd.concat([pd.DataFrame(p_val_all[key]).stack() for key in
                        p_val_all], axis=1)
    new_df.columns = list(p_val_all.keys())
    # new_df = pd.Panel(p_val_all).to_frame()
    return new_df


class MyTboard(callbacks.TensorBoard):

    def __init__(self, args):
        super(MyTboard, self).__init__(args)
        self.dum_step = 0

    def on_epoch_end(self, net, **kwargs):

        out_feat = net.forward(kwargs['dataset_train'], training=False)
        pred = out_feat[0].argmax(1)

        tr_idx = kwargs['dataset_train'].idx
        val_idx = kwargs['dataset_train'].val_idx
        ts_idx = (set(np.arange(2727)) - set(tr_idx)) - set(val_idx)
        ts_idx = np.array(list(ts_idx))
        y_true = kwargs['dataset_train'].y
        tr_acc = accuracy_score(y_true[tr_idx], pred[tr_idx])
        val_acc = accuracy_score(y_true[val_idx], pred[val_idx])
        ts_acc = accuracy_score(y_true[ts_idx], pred[ts_idx])

        self.writer.add_scalar('tr_acc', tr_acc, self.dum_step)
        self.writer.add_scalar('val_acc', val_acc, self.dum_step)
        self.writer.add_scalar('test_acc', ts_acc, self.dum_step)
        self.dum_step += 1

        # Visualize output
        self.writer.add_embedding(out_feat[4][val_idx],
                                  metadata=list(y_true[val_idx]),
                                  global_step=self.dum_step,
                                  tag='gnn_output')
        self.writer.add_embedding(out_feat[5][val_idx],
                                  metadata=list(y_true[val_idx]),
                                  global_step=self.dum_step,
                                  tag='input')

        super(MyTboard, self).on_epoch_end(net, **kwargs)


def val_score(net, ds, y=None):
    """Validation score function for stoppping criterion"""
    # Classification loss using validation set only in transductive setting
    y_pred = net.forward(ds, training=False)
    val_idx = ds.val_idx
    val_gt_target = y[val_idx].argmax(1)
    val_pred_target = y_pred[val_idx].detach().numpy().argmax(1)
    return accuracy_score(val_gt_target, val_pred_target)


def mgmc_val_score(net, ds, y=None):
    """Validation score function for stoppping criterion"""
    # Classification loss using validation set only in transductive setting
    y_pred = net.forward(ds, training=False)
    num_class = y.shape[-1]
    pred_matrix = y_pred[0]
    # pred_features = pred_matrix[:, :-num_class]
    pred_target = pred_matrix[:, -num_class:]
    val_idx = ds.val_idx
    val_gt_target = y[val_idx].argmax(1)
    val_pred_target = pred_target[val_idx].detach().numpy().argmax(1)
    return accuracy_score(val_gt_target, val_pred_target)


def cv_search_scorer(gt_target, pred_target):
    res = gt_target + pred_target
    return res


def get_thresholded_graph(meta_data_tensor, threshold_list):
    """Get the thresholded graph given meta information and a list of thresholds."""
    adj_list = []
    for k, v in enumerate(threshold_list):
        meta_ = meta_data_tensor[:, k:k+1]
        dist = meta_[:, None, :] - meta_[None, :, :]
        cur_adj = torch.abs(dist) <= v
        cur_adj = cur_adj.long()
        adj_list.append(cur_adj.squeeze())
    cur_adj = torch.stack(adj_list, -1)
    return cur_adj


def get_thresholded_eucledian(input_arr, threshold_list):
    """Get binary adjacency matrix where a connection between a node is based on
    it's eucledian distance to another point in the given input space."""

    # Given numerical data tensor, impute missing data using mean imputation

    data_arr = np.where(np.isnan(input_arr), np.ma.array(input_arr, mask=np.isnan(input_arr)).mean(axis=0), input_arr)
    assert np.isnan(data_arr).all() == False, 'Make sure there are no longer nan in data arr'


    # Then normalize vector
    preprocessor = StandardScaler()
    data_arr = preprocessor.fit_transform(data_arr)

    adj_list = []
    for k, v in enumerate(threshold_list):
        dist = euclidean_distances(data_arr, data_arr)
        dist = torch.tensor(dist, dtype=torch.float32)
        cur_adj = torch.abs(dist) < v
        cur_adj = cur_adj.long()
        adj_list.append(cur_adj.squeeze())
    cur_adj = torch.stack(adj_list, -1)
    cur_adj = torch.tensor(cur_adj, dtype=torch.float32)
    return cur_adj


def get_feature_importance():
    """Get feature importance of given list of pickled sklearn classifiers. """
    model_list = [
        '../examples/outputs/chd_3class/exp_012/model_RandomForestClassifier.pkl',
        '../examples/outputs/chd_3class/exp_012/model_LogisticRegression.pkl',
    ]
    all_importances = {}
    for i in model_list:
        print(i)
        predictor_ = i.split('/')[-1].replace('pkl', '')
        with open(i, 'rb') as fpath:
            cur_model = pickle.load(fpath)
            list_importances = []
            for idx_, j   in enumerate(cur_model):
                model_ = j
                if 'randomforest'in i.lower():
                    res_cols, importances = plot_imp_features_rf(model_._final_estimator.best_estimator_,
                                                                       col_names, top_features=29)
                elif 'logistic'in i.lower():
                    res_cols = col_names
                    # Mean of n logistic regression models
                    importances = np.abs(model_._final_estimator.best_estimator_.coef_).mean(0)

                cur_item = pd.Series(importances, index=res_cols)
                list_importances.append(cur_item)

        all_importances[predictor_] = list_importances


def sound():
    import time
    for i in range(60):
        print('\a')
        time.sleep(1)

def umap2D_and_plot(numpy_arr, label_data, title, class_list):
    """Perform UMAP embedding on given array and plot 2D embedding."""
    latent_embedding = umap.UMAP().fit_transform(numpy_arr)
    fig, ax = plt.subplots()
    scatter = ax.scatter(latent_embedding[:, 0], latent_embedding[:, 1], c=label_data)
    # legend1 = ax.legend(*scatter.legend_elements(),
    #                     loc="best", title="Classes")
    plt.legend(handles=scatter.legend_elements()[0], labels=class_list)
    # ax.add_artist(legend1)
    plt.axis('tight')
    plt.title(title)
    plt.show()


# From https://github.com/facebookresearch/PyTorch-BigGraph/pull/67
def l2_mat(b1, b2):
    """b1 has size B x M x D, b2 has size b2 B x N x D, res has size P x M x N """
    b1_norm = b1.pow(2).sum(dim=-1, keepdim=True)
    b2_norm = b2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(b2_norm.transpose(-2, -1), b1, b2.transpose(-2, -1), alpha=-2).add_(b1_norm)
    # mask = 1.0 - torch.ones(res.shape[0]).diag().to(res.device)
    res = res.clamp_min_(torch.finfo(torch.float32).eps).sqrt_()
    # res = res * mask
    return res

    # res = torch.addmm(x2_norm.transpose(-2, -1), gl_out, gl_out.transpose(-2, -1), alpha=-2).add_(
    #     x1_norm)


def select_p_data_df(df, p_select, random_seed):
    """
    Randomly select a given percentage of data from a dataframe

    Parameters:
        df: pd.Dataframe
            Featue matrix as dataframe
        p_select: float
            In range (0,1)
        random_seed: int
    Return:
        # TODO
    """
    data_arr = df.copy()
    data_arr = np.array(data_arr)
    M_num_entries = np.product(data_arr.shape[:])

    # Current index of known entries
    idx = np.where(~np.isnan(data_arr))
    idx_1 = idx[0]
    idx_2 = idx[1]
    cur_idx_len = idx_1.size

    # Current percentage of known entries
    current_p_known = cur_idx_len / M_num_entries
    # assert p_select < current_p_known, 'Percentage is larger than available ' \
    #                                    'entries'
    print('Percentage of current known entries {}'.format(current_p_known))

    # Set randomly selected entries to nan
    desired_entries = int(cur_idx_len * p_select)
    remove_entries = cur_idx_len - desired_entries

    np.random.seed(random_seed)
    idx_choice = np.random.choice(cur_idx_len, remove_entries, replace=False)
    new_idx = idx_1[idx_choice], idx_2[idx_choice]
    data_arr[new_idx] = np.nan

    # New percentage of entries
    new_p_known = np.where(~np.isnan(data_arr))[0].size / M_num_entries
    print('New percentage of known entries {}'.format(new_p_known))

    return data_arr, new_idx


def pretty_table(pkl_list):
    """
    Given a list of pickle file. Create a dataframe for all results in that pickle file.
    """
    all_df = []
    for p in pkl_list:
        print(p)
        with open(p, 'rb') as pkl:
            cur_data = pickle.load(pkl)
            cur_res_df = pretty_metrics(cur_data)
            cur_res_df['model'] = p.split('/')[-1].replace('.pkl', '')
        all_df.append(cur_res_df)
    return all_df


def pretty_metrics(gmc_output):
    """
    Given a list which contains a tuple of (ground truth, sigmoid_output,
    test_pred, test_RMSE) calculate, ROC-AUC, accuracy, f1-score. Then return
    classification and imputation metrics as one table.
    """
    # For every experiment calculate classification metrics
    pretty_output_list = []
    p_list = [0.25, 0.5, .75, 1.]
    if len(gmc_output[0][0][0].shape) > 1 and gmc_output[0][0][0].shape[-1] > 1:
        cur_data = gmc_output[0][0][0]
        num_class = cur_data.shape[-1]
    else:
        num_class = 1
    print(num_class)
    for k, experiment in enumerate(gmc_output):
        cur_auc = [roc_auc_score(x[0], x[1]) for x in
                   experiment]
        if num_class > 1:
            print('num_class here is %d' %num_class)
            cur_acc = [accuracy_score(np.argmax(x[0], 1), x[2]) for x in
                       experiment]
            cur_f1 = [f1_score(np.argmax(x[0], 1), x[2], average='weighted')
                      for x in experiment]

            # Weighted average of recall and precision
            clf_report = \
                [classification_report(np.argmax(x[0],1), x[2],
                                       output_dict=True) for x in experiment]
            clf_report = [pd.DataFrame(x) for x in clf_report]

            cur_recall = [x.T for x in clf_report]
            cur_recall = [x.recall['weighted avg'] for x in cur_recall]
            cur_specificity = [x.T for x in clf_report]
            cur_specificity = [x.precision['weighted avg'] for x in
                               cur_specificity]
        else:
            cur_acc = [accuracy_score(x[0], x[2]) for x in experiment]
            cur_f1 = [f1_score(x[0], x[2]) for x in experiment]
            cur_recall = [pd.DataFrame(classification_report(x[0], x[2], \
                                                             output_dict=True)).T.recall[1] for x in experiment]

            cur_specificity = [pd.DataFrame(classification_report(x[0], x[2],
                                                                  output_dict=True)).T.recall[0] for x in experiment]
        cur_rmse = [x[3] for x in experiment]


        cur_append = [cur_auc, cur_acc, cur_rmse, cur_f1, cur_recall, cur_specificity]
        cur_append = pd.DataFrame(cur_append).T
        cur_append.columns = ['AUC', 'Accuracy','RMSE', 'F-measure',
                              'Sensitivity', 'Specificity']
        cur_append['%'] = p_list[k]
        pretty_output_list.append(cur_append)
    res_df = pd.concat(pretty_output_list)
    return res_df


def get_model_names(path_list):
    """Get model name given list of paths."""
    res_list = []
    for i in path_list:
        cur_name = i.split('/')[-1]
        cur_name = cur_name.replace('output_pred_', '').replace('_.pkl', '')
        res_list.append(cur_name)
    return res_list


def quick_load_pkl(pkl_file):
    """Load pickplot_imp_features_rfle file"""
    # Load gmc output numpy for every experiment
    # pkl_file = './gmc_logs/gmc_res_prior_graph.pkl'
    with open(pkl_file, 'rb') as f_pkle:
        gmc_output = pickle.load(f_pkle)
    return gmc_output


def get_summary_statistics(df, cur_index, len_numerical, binary_idx_list):
    """Get summary statistics of given indices from dataframe."""
    summary_stat_list = []
    df = df.copy()
    for i in cur_index:
        # Summary stat using mean and std for numerical data
        # print(cur_grouped_df_.sum())
        if i < len_numerical:
            cur_grouped_df_ = df[[i, 'label']].astype(float)
            cur_grouped_df_.dropna(inplace=True)
            cur_grouped_df_ = cur_grouped_df_.groupby('label')
            cur_mean_ = cur_grouped_df_.mean().round(2).astype(str)
            cur_std_ = cur_grouped_df_.std().round(2).astype(str)
            res_ = cur_mean_ + u" \u00B1 " +  cur_std_
        # Percentage for non-numerical data
        else:
            cur_df = df[[i, 'label']]
            num_class_ = cur_df.label.unique().size
            cur_df.replace(0.5, np.nan, inplace=True)
            cur_df.dropna(inplace=True)
            cur_count_ = pd.Series(cur_df.groupby([i, 'label']).size().to_numpy()[-num_class_:])
            cur_sum_ = cur_df.groupby('label').size()
            # cur_grouped_df_ = cur_df[cur_df[cur_df.columns[0]] == 1].groupby('label')
            # cur_count_ = cur_grouped_df_.count()
            # print(f"Current count {cur_count_}")
            # mask_ = (cur_df[cur_df.columns[0]] == 1) | (cur_df[cur_df.columns[0]] == 0)
            # cur_sum_ = cur_df.groupby('label').count()
            # print(f"Current sum {cur_sum_}")
            percentage_ = ((cur_count_/cur_sum_) * 100).round(1)
            res_ = percentage_.astype(str)+"%"
        res_ = pd.DataFrame(res_).T
        # print(res_)
        summary_stat_list.append(res_)
    return summary_stat_list


def get_group_statistics(cur_selected_df, cur_index, len_numerical):
    """Given a dataframe of feature and column label get group-wise comparison using statistical testing."""
    cur_selected_df = cur_selected_df.copy()

    p_val_list = []
    is_paired = False
    for i in cur_index:
        print(f"Calculating {i}")
        cur_labels_ = cur_selected_df.label.unique()
        len_class_ = cur_labels_.size
        is_numerical = i < len_numerical
        if len_class_ == 2:
            print(i, "Two class labels")
            group_1 = cur_selected_df[i][cur_selected_df.label == cur_labels_[0]]
            group_2 = cur_selected_df[i][cur_selected_df.label == cur_labels_[1]]

            if is_numerical:
                print(i, "is_numerical")
                # Check for normality
                is_gaussian = (shapiro(group_1)[1] > 0.05) and (shapiro(group_2)[0] > 0.05)

                # Two group distribution comparison
                if is_gaussian:
                    print("gaussian")
                    stat_test = ttest_rel if is_paired else ttest_ind
                else:
                    print("not gaussian")
                    stat_test = wilcoxon if is_paired else mannwhitneyu
                p_val = stat_test(group_1, group_2)[1]

            else:
                # Chi-square test
                print(i, "not numerical")
                cur_df = cur_selected_df[[i, 'label']]
                cur_df.replace(0.5, np.nan, inplace=True)
                cur_df.dropna(inplace=True)
                print(cur_df)
                print(cur_df.groupby('label').count())
                cur_df.rename(columns={i: "{}".format(i)}, inplace=True)
                # _, _, stats = pg.chi2_independence(cur_df[[str(i), 'label']].dropna(), str(i), 'label')
                _, _, stats = pg.chi2_independence(cur_df, str(i), 'label')
                print(stats)
                p_val = stats[stats.test == 'pearson'].pval.item()

        else:
            print(i, "More than two class labels")

            if is_numerical:
                print(i, 'is numerical')
                # Get all current groups
                cur_df = cur_selected_df[[i, 'label']]
                cur_df.dropna(inplace=True)
                group_list = [cur_df[i][list(cur_df.label == v)] for k, v in enumerate(cur_labels_)]
                group_list = [np.array(x) for x in group_list]


                # Test for normality
                all_gauss_list = [shapiro(x)[1] > 0.05 for x in group_list]
                are_all_gaussian = all(all_gauss_list)

                # Parameteric variance test - ANOVA
                if are_all_gaussian:

                    if is_paired:
                        raise NotImplementedError('Repeated Measures ANOVA currently not supported...')

                    stat_test = f_oneway

                # Non-parameteric variance test
                else:
                    stat_test = friedmanchisquare if is_paired else kruskal
                # print(group_list)
                p_val = stat_test(*group_list)[1]
            else:
                # Chi-square test
                print(i, "not numerical")
                cur_df = cur_selected_df[[i, 'label']]
                print(cur_df)
                cur_df.replace(0.5, np.nan, inplace=True)
                cur_df.dropna(inplace=True)
                cur_df.rename(columns={i: "{}".format(i)}, inplace=True)
                # _, _, stats = pg.chi2_independence(cur_df[[str(i), 'label']].dropna(), str(i), 'label')
                _, _, stats = pg.chi2_independence(cur_df, str(i), 'label')
                p_val = stats[stats.test == 'pearson'].pval.item()

        print(f"{i} -> p-val is {p_val}")
        if p_val is not None:
            p_val = modify_p_val(p_val)
            print(f"{i} -> p-val after modify {p_val}")
        p_val_list.append(p_val)

    return p_val_list


def get_index(all_column_names_list, feature_name_list):
    """Get index of current importance features"""
    res_ = [all_column_names_list.index(x) for x in feature_name_list]
    return res_


def modify_p_val(cur_p):
    """Convert p-value to use <0.05, <0.005, or <0.0005 if value less than 0.05, 0.005, 0.0005 respectively. Else
    just use rounded p-value."""
    ret_val = None
    for k, val_ in enumerate([0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]):
        if cur_p < val_:
            ret_val = '<{}'.format(val_)
        elif cur_p == val_:
            ret_val = '{}'.format(val_)
    if ret_val is None:
        ret_val = '{}'.format(np.round(cur_p, decimals=3))
    return ret_val


def get_binary_indices(data_arr, idx_list):
    """
    Given data array and non-numerical indices, get indices of binary features. This information can be use to impute
    missing to 0.5"""
    binary_var_list = []
    for i in idx_list:
        cur_unique = pd.DataFrame(data_arr[:, i]).drop_duplicates().dropna()
        cur_unique_len = cur_unique.size
        if cur_unique_len == 2:
            # print(i, cur_unique[0].tolist())
            binary_var_list.append(i)
    return binary_var_list


# def impute_binary(data_arr, indices_list, impute_val=0.5):
#     """Given data array and indices impute binary variable with impute_val."""
#     for i in indices_list:
#         cur_arr_ = data_arr[:, i]
#         if isinstance(cur_arr_, np.ndarray):
#             cur_arr_ = np.nan_to_num(cur_arr_, impute_val)
#         data_arr[:, i] = cur_arr_
#     return data_arr


def take_first_col_and_impute(data_arr):
    """
    Just take the first column in array and impute nan with 0.5. Used within ColumnTransformer.

    :param data_arr:
    :return:
    """
    for i in range(data_arr.shape[-1]):
        lb = LabelBinarizer()
        cur_data = lb.fit_transform(data_arr[:, i])
        if cur_data.shape[-1] == 3:
            assert lb.classes_[-1] == "zzzzz"
            mask_ = cur_data[:, -1].astype(bool)
            cur_arr = cur_data[:, :1]
            cur_arr = cur_arr.astype(np.float32)
            cur_arr[mask_] = 0.5
        else:
            cur_arr = cur_data.astype(np.float32)
        data_arr[:, i:i+1] = cur_arr

    return data_arr


def encode_ordinal_feat(df, column_list, encoding_dict):
    """
    Change encoding of ordinal features.

    """
    df_ = df.copy()
    for i in column_list:
        if i in df_.columns:
            cur_val = df[[i]]
            for k, v in encoding_dict.items():
                cur_val = cur_val.replace(k, v)
            df_[[i]] = cur_val
    return df_


def get_train_test_dataset_list(args, trainer, pre_processor, is_mgmc=False):
    """Get train test dataset which will be used as input to calculate feature attributions."""

    # Get dataset
    dataset = trainer.data_provider
    cv = StratifiedKFold(n_splits=args.cv_folds, random_state=args.rand_seed)

    folds_list = cv.split(dataset.data_x, dataset.data_y.argmax(1))
    folds_list = list(folds_list)
    train_x_list = []
    train_y_list = []
    test_x_list = []
    test_y_list = []

    if is_mgmc:
        inner_cv = StratifiedKFold(n_splits=5, random_state=args.random_state)
        args.p_remain = 1.0
        args.len_numerical_features = len(dataset.numerical_idx)
        args.num_class = np.unique(dataset.data_y.argmax(1)).size
        torch.backends.cudnn.enabled=False

    for folds in folds_list:
        train_index = folds[0]
        test_index = folds[1]

        if is_mgmc:
            datatrainx = dataset.data_x
            datatestx = dataset.data_x
            datatrainy = dataset.data_y
            datatesty = dataset.data_y

            tr_split = inner_cv.split(train_index, dataset.data_y[train_index].argmax(1))
            tr_idx, val_idx = list(tr_split)[0]
            val_index = train_index[val_idx]
            train_index = train_index[tr_idx]

        else:
            datatrainx = dataset.data_x[train_index, :]
            datatestx = dataset.data_x[test_index, :]
            datatrainy = dataset.data_y[train_index]
            datatesty = dataset.data_y[test_index]


        #     nonun_preprocessor = Pipeline([('preprocessing', preprocessor)])
        nonnum_features = pre_processor.fit_transform(dataset.data_x)[:, len(dataset.numerical_idx):]

        steps = list()
        steps.append(("preprocessor", pre_processor))
        to_float_ = FunctionTransformer(as_float)
        steps.append(('to_float', to_float_))
        pipe = Pipeline(steps)

        datatestx = pipe.fit_transform(datatestx)
        datatrainx = pipe.fit_transform(datatrainx)

        # Combine numerical and non-numerical features
        if is_mgmc:
            nonnum_train_feat = nonnum_features[:, :]
            nonnum_test_feat = nonnum_features[:, :]
        else:
            nonnum_train_feat = nonnum_features[train_index, :]
            nonnum_test_feat = nonnum_features[test_index, :]
        train_data_x = datatrainx[:, :len(dataset.numerical_idx)]
        test_data_x = datatestx[:, :len(dataset.numerical_idx)]
        train_data_x = np.concatenate([train_data_x, nonnum_train_feat], 1).astype(np.float32)
        test_data_x = np.concatenate([test_data_x, nonnum_test_feat], 1).astype(np.float32)
        if is_mgmc:
            #         train_data_x = np.concatenate([train_data_x, datatrainy], 1)
            #         test_data_x = np.concatenate([test_data_x, datatesty], 1)
            preprocessor_pipe = Pipeline([("preprocessor", pre_processor)])
            mask_matrix = trainer.get_nonnum_observed_mask_matrix(train_data_x, preprocessor_pipe)
            mask_matrix = np.concatenate([mask_matrix, np.zeros_like(datatesty).astype(bool)], 1)
            train_adj = dataset.adj
            test_adj = dataset.adj
            train_dataset = dp.TransductiveMGMCDataset(train_data_x, train_adj, datatrainy,
                                                       train_index, val_index, mask_matrix, is_training=True,
                                                       args=args)
            test_dataset = dp.TransductiveMGMCDataset(test_data_x, test_adj, datatesty,
                                                      test_index, val_index, mask_matrix, is_training=False,
                                                      args=args)
            train_x_list.append(train_dataset)
            train_y_list.append(datatrainy)
            test_x_list.append(test_dataset)
            test_y_list.append(datatesty)

        else:
            train_x_list.append(train_data_x)
            train_y_list.append(datatrainy)
            test_x_list.append(test_data_x)
            test_y_list.append(datatesty)

    return train_x_list, train_y_list, test_x_list, test_y_list


def get_feature_attributions(args, model_list, data_x_list, data_y_list, col_names):
    """Get feature attributions for given list of saved Pipeline models and output dataset lists from
    get_train_test_dataset_list()."""

    list_importances = []
    for k, i in enumerate(model_list):

        print("Calculating feature importances for Model {}".format(k))
        # Get model
        #     model = i._final_estimator.best_estimator_.module_
        device_ = "cuda:1"
        if "MGMC" in i._final_estimator.module_.__class__.__name__:
            model = i._final_estimator.module_
            model.to(device_)
            targets = torch.tensor(data_y_list[k].argmax(1)).to(torch.device(device_))
            data_loader = torch.utils.data.DataLoader(data_x_list[k], batch_size=data_x_list[0].feat_data.shape[0])

            for batch in data_loader:
                batch[0] = [x.to(device_) for x in batch[0]]
            input_ = batch[0]

            def model_forward(feat_matrix, input_list):
                format_input = [feat_matrix] + input_list
                out = model(format_input)
                out = out[0][:, -args.num_class:]
                return out
            attr = IntegratedGradients(model_forward, )
            fa_list = []
            # for class_i in range(args.num_class):
            #     cur_fa = attr.attribute(input_[0], target=class_i, additional_forward_args=input_[1:])
            #     fa_list.append(torch.abs(cur_fa))
            # fa_train = torch.stack(fa_list, -1).sum(-1)
            # mean_fa = fa_train.mean(0)
            fa_train = attr.attribute(input_[0], target=targets, additional_forward_args=input_[1:])
            fa_train = torch.abs(fa_train)
            mean_fa = fa_train.mean(0)
            mean_fa = mean_fa[:-args.num_class]

        else:
            model = i._final_estimator.module_
            model.to(device_)
            attr = IntegratedGradients(model)
            input_ = torch.tensor(data_x_list[k]).to(torch.device(device_))
            targets = torch.tensor(data_y_list[k].argmax(1)).to(torch.device(device_))
            baseline = input_ * 0.
            fa_train = attr.attribute(input_, baseline, target=targets)
            fa_train = torch.abs(fa_train)
            mean_fa = fa_train.mean(0)
        mean_fa = mean_fa.cpu().numpy()
        cur_item = pd.Series(mean_fa, index=col_names)
        list_importances.append(cur_item)
    return list_importances


def save_fig_list(out_path, fig_list):
    """Save given list of figures"""
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in fig_list:
        save_fname = "{}_cm.pdf".format(i.get_axes()[0].get_title())
        save_fname = os.path.join(out_path, save_fname)
        print("Saving {}".format(save_fname))
        i.savefig(save_fname, format='pdf', bbox_inches='tight')
