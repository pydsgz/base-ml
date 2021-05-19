import numpy as np

from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, accuracy_score
from skorch import NeuralNetClassifier
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch_geometric as tg
from torch_geometric.nn import GATConv, ChebConv, SAGEConv, GCNConv

from base_ml import utils
from base_ml import layers


class ConfigMixin:
    def load_config(self):
        conf_path = self.args.__dict__.get('model_conf_path', None)
        if conf_path is not None:
            model_config = OmegaConf.load(conf_path)
        else:
            conf_path = './config/{}/{}.yaml'.format(self.args.dataset, self.__class__.__name__)
            model_config = OmegaConf.load(conf_path)
        return model_config


class OutputLatentMixin:
    def get_latent_embedding(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        for k, net in enumerate(self.gnn_layers):
            x = net(x, edge_idx)
            x = F.relu(x)
        return x


class DeepMLPRegressor(nn.Module):
    """ Five-layer MLP model. """

    def __init__(self, hid_dim1=10, hid_dim2=10, hid_dim3=8, hid_dim4=8,
                 hid_dim5=8, p_dropout=0.3):
        super(DeepMLPRegressor, self).__init__()
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.hid_dim3 = hid_dim3
        self.hid_dim4 = hid_dim4
        self.hid_dim5 = hid_dim5
        self.p_dropout = p_dropout
        self.mlp = nn.Sequential(
            nn.Linear(24, hid_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim1),
            nn.Dropout(p_dropout),
            nn.Linear(hid_dim1, hid_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim2),
            nn.Dropout(p_dropout),
            nn.Linear(hid_dim2, hid_dim3),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim3),
            nn.Dropout(p_dropout),
            nn.Linear(hid_dim3, hid_dim4),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim4),
            nn.Dropout(p_dropout),
            nn.Linear(hid_dim4, hid_dim5),
            nn.ReLU(),
            nn.BatchNorm1d(hid_dim5),
            nn.Dropout(p_dropout)
        )
        self.out = torch.nn.Linear(hid_dim5, 1)

    def forward(self, x, **kwargs):
        """ #TODO """
        x = x.float().to(**kwargs)
        x = self.mlp(x)
        x = self.out(x)
        return x

    @staticmethod
    def scoring_loss(y_true, y_pred):
        se = (y_true - y_pred) ** 2
        mse = se.mean()
        # rmse = torch.sqrt(mse)
        return mse


class DeepMLPClassifier(torch.nn.Module):
    """ Five-layer MLP model. """

    def __init__(self, input_dim=24, hid_dim1=64, hid_dim2=32, hid_dim3=8,
                 p_dropout=0.3, n_class=3):
        super(DeepMLPClassifier, self).__init__()
        self.n_class = n_class
        self.input_dim = input_dim
        self.hid_dim1 = hid_dim1
        self.hid_dim2 = hid_dim2
        self.hid_dim3 = hid_dim3

        self.p_dropout = p_dropout

        self.enc = nn.Sequential(
            # nn.Dropout(p_dropout),
            nn.Linear(input_dim, hid_dim1),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(hid_dim1),
            nn.ReLU(),

            nn.Linear(hid_dim1, hid_dim2),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(hid_dim2),
            nn.ReLU(),

            nn.Linear(hid_dim2, hid_dim3),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(hid_dim3),
            nn.ReLU(),
        )

        self.dec = nn.Sequential(
            # nn.Dropout(p_dropout),
            nn.Linear(hid_dim3, hid_dim2),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(hid_dim2),
            nn.ReLU(),

            nn.Linear(hid_dim2, hid_dim1),
            nn.Dropout(p_dropout),
            nn.BatchNorm1d(hid_dim1),
            nn.ReLU(),
            nn.Linear(hid_dim1, input_dim))

        self.out = torch.nn.Linear(hid_dim3, n_class)

    def forward(self, x, **kwargs):
        """ #TODO """
        pred_x = x.float().to(**kwargs)
        enc_x = self.enc(pred_x)
        recon_x = self.dec(enc_x)
        # pred_x = self.out(torch.cat([enc_x, x], 1))
        pred_x = self.out(enc_x)
        pred_x = F.softmax(pred_x, dim=-1)
        return pred_x, recon_x, x


class MLPClassifier(nn.Module, ConfigMixin):
    """Fully-connected Feed-Forward Neural Networks aka Multilayer Perceptrons with non-linear activation functions."""

    def __init__(self, args, layer_list=None, p_list=None):
        super(MLPClassifier, self).__init__()
        self.args = args
        class_kwargs = self.load_config()

        # hyperparameters from hp search
        if layer_list is not None:
            cur_layers = list(class_kwargs.layers)
            cur_layers[1:-1] = [layer_list] * len(cur_layers[1:-1])
            class_kwargs.layers = cur_layers
        if p_list is not None:
            class_kwargs.p = [p_list] * len(class_kwargs.p)

        layers_list = list(class_kwargs.get('layers'))
        dropout_list = list(class_kwargs.get('p'))
        mlp_layers = []
        for k, v in enumerate(layers_list[:-2]):
            cur_layers_ = [
                nn.Linear(layers_list[k], layers_list[k + 1]),
                nn.Dropout(dropout_list[k]),
                nn.ReLU(),
                nn.BatchNorm1d(layers_list[k + 1])
            ]
            mlp_layers.extend(cur_layers_)

        self.enc = nn.Sequential(*mlp_layers)
        self.out = torch.nn.Linear(layers_list[-2], layers_list[-1])

    def forward(self, x):
        enc_x = self.enc(x)
        pred_x = self.out(enc_x)
        pred_x = F.softmax(pred_x, dim=-1)
        return pred_x


class LGNN(torch.nn.Module):
    """ n-layer GAT and fully connected output layer. """

    def __init__(self, args, gnn_layers):
        super(LGNN, self).__init__()
        self.args = args
        layers_ = gnn_layers
        gl_layers = [3, 6, 3]
        gat_layers = []
        lin_layers = []

        for k, v in enumerate(gl_layers[:-2]):
            in_layer = gl_layers[k]
            out_layer = gl_layers[k+1]
            cur_layer = nn.Linear(in_layer, out_layer)
            gat_layers.append(cur_layer)
        self.gat_layers = torch.nn.ModuleList(gat_layers)

        for k, v in enumerate(layers_[:-2]):
            cur_layer = nn.Linear(layers_[k], layers_[k + 1])
            lin_layers.append(cur_layer)
        self.lin_layers = torch.nn.ModuleList(lin_layers)

        out_list = []
        num_incept = 3
        for i in range(num_incept):
            # out_list.append(ChebConv(layers_[-2] + layers_[-2], layers_[-1],
            #                          K=i+1))
            out_list.append(ChebConv(layers_[-2], layers_[-1],
                                     K=(i+1)*3))
            # out_list.append(GATConv(layers_[-2], layers_[-1],
            #                          heads=3, concat=True))
        self.out_list = torch.nn.ModuleList(out_list)

        final_inp_dim = (layers_[-1] * num_incept)
        self.last = nn.Linear(final_inp_dim, layers_[-1])

        # self.out = ChebConv(layers_[-2] + layers_[0], layers_[-1], K=3)
        self.temp = torch.nn.Parameter(
            torch.tensor(1e-1, requires_grad=True).cuda())
        self.theta = torch.nn.Parameter(torch.ones(2727,
                                                   requires_grad=True).cuda())

    def forward(self, data_x):
        x = data_x[5]
        x2 = data_x[0]
        x = x.float()

        edge_idx = data_x[1]
        edge_idx = edge_idx[:, data_x[2]]
        edge_idx = edge_idx.nonzero().t().contiguous()

        # Transformation to latent space
        for k, (net, net2) in enumerate(zip(self.gat_layers, self.lin_layers)):
            x, x2 = net(x), net2(x2)
            x, x2 = F.relu(x), F.relu(x2)

        # Graph learning
        diff = x.unsqueeze(1) - x.unsqueeze(0)
        dist = -torch.pow(diff, 2).sum(2)

        # temp = 1.0 + self.temp
        theta = 2.0
        prob_matrix = torch.exp(dist/theta)
        # print(prob_matrix)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(prob_matrix)

        last_layer = []
        for out in self.out_list:
            cur_out = out(x2, edge_idx,
                          edge_weight=edge_wts)
            last_layer.append(cur_out)

        last_out = torch.cat(last_layer, -1)
        x = self.last(last_out)
        x = F.softmax(x, dim=-1)

        val_idx = data_x[4][0]
        tr_idx = data_x[3][0]
        return x, tr_idx, val_idx, prob_matrix, last_out, data_x[0]


class MGMCCheby(torch.nn.Module, ConfigMixin):
    """# TODO MGMC Paper"""
    def __init__(self, args, hidden_dim=None, timesteps=None, K=None):
        super(MGMCCheby, self).__init__()
        rnn_list = []
        gnn_list = []
        fc_list = []
        self.args = args
        conf = self.load_config()
        self.timesteps = conf.class_kwargs.get('timesteps', 5)

        # Update config hyperparams so that we can use hyperparameter search algorithms
        if timesteps is not None:
            conf.class_kwargs.timesteps = timesteps
        if K is not None:
            conf.gnn_kwargs.K = K
        if hidden_dim is not None:
            conf.gnn_kwargs.out_channels = hidden_dim
            conf.rnn_kwargs.input_size = hidden_dim
            conf.rnn_kwargs.hidden_size = hidden_dim
            conf.fc_kwargs.in_features = hidden_dim

        for layer in range(args.num_meta):
            gnn_list.append(tg.nn.ChebConv(**conf.gnn_kwargs))
            rnn_list.append(nn.LSTM(**conf.rnn_kwargs, batch_first=True)),
            fc_list.append(nn.Linear(**conf.fc_kwargs))
        self.gnn_list = nn.ModuleList(gnn_list)
        self.rnn_list = nn.ModuleList(rnn_list)
        self.fc_list = nn.ModuleList(fc_list)

        # Aggregation layer
        aggregator_ = conf.class_kwargs.get('aggregator', None)
        sa_kwargs = conf.get('sa_kwargs', None)
        if aggregator_ == "self-attention":
            self.aggregator = layers.MultiHeadAttention(**sa_kwargs)
        else:
            self.aggregator = None

    def forward(self, input_tuple):
        x, adj = input_tuple[0], input_tuple[1]

        rgcn_output = []
        z_bar_list = []

        # Indices to use to distinguish numerical from non-numerical features
        start_idx = self.args.len_numerical_features
        end_idx = self.args.num_class

        for k, (gnn, rnn) in enumerate(zip(self.gnn_list, self.rnn_list)):
            cur_adj = adj[:, :, k]
            cur_adj = cur_adj.nonzero().t().contiguous()
            out_list = []
            for _ in range(self.timesteps):
                gnn_out = gnn(x, cur_adj)
                out_list.append(gnn_out)
            out_tensor = torch.stack(out_list, 1)
            cur_x, _ = rnn(out_tensor)
            cur_x = F.tanh(self.fc_list[k](cur_x))
            rgcn_output.append(cur_x[:, -1, :])  # Take the last timestep's output

            # Output delta + x of current graph. Store and return for dirichlet norm calculation
            cur_z_delta = cur_x[:, -1, :]
            z_bar = x + cur_z_delta
            z_bar[:, start_idx:-end_idx] = torch.sigmoid(z_bar[:, start_idx:-end_idx])
            z_bar_list.append(z_bar)

        # Aggregated outputs of multiple RGCNs
        if self.aggregator is None:
            delta_z = torch.stack(rgcn_output, -1).mean(-1)
        else:
            delta_z = torch.stack(rgcn_output, 1)
            agg_delta_z, attn = self.aggregator(delta_z, delta_z, delta_z)
            delta_z = agg_delta_z.sum(1)
        res_z_bar = torch.stack(z_bar_list, -1)  # output of every graph
        pred_matrix = x + delta_z

        # Apply sigmoid on non-numerical features
        pred_matrix[:, start_idx:-end_idx] = torch.sigmoid(pred_matrix[:, start_idx:-end_idx])
        # z_delta_bar = torch.stack(rgcn_output, -1)
        return pred_matrix, res_z_bar, adj


class MGMCGCN(torch.nn.Module):
    # TODO
    def __init__(self, args, hidden_dim=None, timesteps=None):
        super(MGMCGCN, self).__init__()
        rnn_list = []
        gnn_list = []
        fc_list = []
        self.args = args
        conf = self.load_config()
        self.timesteps = conf.class_kwargs.get('timesteps', 5)

        # Update config hyperparams so that we can use hyperparameter search algorithms
        if timesteps is not None:
            conf.class_kwargs.timesteps = timesteps
        if hidden_dim is not None:
            conf.gnn_kwargs.out_channels = hidden_dim
            conf.rnn_kwargs.input_size = hidden_dim
            conf.rnn_kwargs.hidden_size = hidden_dim
            conf.fc_kwargs.in_features = hidden_dim

        for layer in range(args.num_meta):
            gnn_list.append(tg.nn.GCNConv(**conf.gnn_kwargs))
            rnn_list.append(nn.LSTM(**conf.rnn_kwargs, batch_first=True)),
            fc_list.append(nn.Linear(**conf.fc_kwargs))
        self.gnn_list = nn.ModuleList(gnn_list)
        self.rnn_list = nn.ModuleList(rnn_list)
        self.fc_list = nn.ModuleList(fc_list)

    def forward(self, input_tuple):
        x, adj = input_tuple[0], input_tuple[1]

        rgcn_output = []
        z_bar_list = []

        # Indices to use to distinguish numerical from non-numerical features
        start_idx = self.args.len_numerical_features
        end_idx = self.args.num_class

        for k, (gnn, rnn) in enumerate(zip(self.gnn_list, self.rnn_list)):
            cur_adj = adj[:, :, k]
            cur_adj = cur_adj.nonzero().t().contiguous()
            out_list = []
            for _ in range(self.timesteps):
                gnn_out = gnn(x, cur_adj)
                out_list.append(gnn_out)
            out_tensor = torch.stack(out_list, 1)
            cur_x, _ = rnn(out_tensor)
            cur_x = F.tanh(self.fc_list[k](cur_x))
            rgcn_output.append(cur_x[:, -1, :])  # Take the last timestep's output

            # Output delta + x of current graph. Store and return for dirichlet norm calculation
            cur_z_delta = cur_x[:, -1, :]
            z_bar = x + cur_z_delta
            z_bar[:, start_idx:-end_idx] = torch.sigmoid(z_bar[:, start_idx:-end_idx])
            z_bar_list.append(z_bar)

        delta_z = torch.stack(rgcn_output, -1).mean(-1)
        res_z_bar = torch.stack(z_bar_list, -1) # output of every graph
        pred_matrix = x + delta_z

        # Apply sigmoid on non-numerical features
        pred_matrix[:, start_idx:-end_idx] = torch.sigmoid(pred_matrix[:, start_idx:-end_idx])
        # z_delta_bar = torch.stack(rgcn_output, -1)
        return pred_matrix, res_z_bar, adj


class MGMCGAT(torch.nn.Module):
    # TODO
    def __init__(self, args, hidden_dim=None, timesteps=None):
        super(MGMCGAT, self).__init__()
        rnn_list = []
        gnn_list = []
        fc_list = []
        self.args = args
        conf = self.load_config()
        self.timesteps = conf.class_kwargs.get('timesteps', 5)

        # Update config hyperparams so that we can use hyperparameter search algorithms
        if timesteps is not None:
            conf.class_kwargs.timesteps = timesteps
        if hidden_dim is not None:
            conf.gnn_kwargs.out_channels = hidden_dim
            conf.rnn_kwargs.input_size = hidden_dim
            conf.rnn_kwargs.hidden_size = hidden_dim
            conf.fc_kwargs.in_features = hidden_dim

        for layer in range(args.num_meta):
            gnn_list.append(tg.nn.GATConv(**conf.gnn_kwargs))
            rnn_list.append(nn.LSTM(**conf.rnn_kwargs, batch_first=True)),
            fc_list.append(nn.Linear(**conf.fc_kwargs))
        self.gnn_list = nn.ModuleList(gnn_list)
        self.rnn_list = nn.ModuleList(rnn_list)
        self.fc_list = nn.ModuleList(fc_list)

    def forward(self, input_tuple):
        x, adj = input_tuple[0], input_tuple[1]

        rgcn_output = []
        z_bar_list = []

        # Indices to use to distinguish numerical from non-numerical features
        start_idx = self.args.len_numerical_features
        end_idx = self.args.num_class

        for k, (gnn, rnn) in enumerate(zip(self.gnn_list, self.rnn_list)):
            cur_adj = adj[:, :, k]
            cur_adj = cur_adj.nonzero().t().contiguous()
            out_list = []
            for _ in range(self.timesteps):
                gnn_out = gnn(x, cur_adj)
                out_list.append(gnn_out)
            out_tensor = torch.stack(out_list, 1)
            cur_x, _ = rnn(out_tensor)
            cur_x = F.tanh(self.fc_list[k](cur_x))
            rgcn_output.append(cur_x[:, -1, :])  # Take the last timestep's output

            # Output delta + x of current graph. Store and return for dirichlet norm calculation
            cur_z_delta = cur_x[:, -1, :]
            z_bar = x + cur_z_delta
            z_bar[:, start_idx:-end_idx] = torch.sigmoid(z_bar[:, start_idx:-end_idx])
            z_bar_list.append(z_bar)

        delta_z = torch.stack(rgcn_output, -1).mean(-1)
        res_z_bar = torch.stack(z_bar_list, -1) # output of every graph
        pred_matrix = x + delta_z

        # Apply sigmoid on non-numerical features
        pred_matrix[:, start_idx:-end_idx] = torch.sigmoid(pred_matrix[:, start_idx:-end_idx])
        # z_delta_bar = torch.stack(rgcn_output, -1)
        return pred_matrix, res_z_bar, adj


class InceptionGCN(torch.nn.Module, ConfigMixin):
    def __init__(self, args):
        super(InceptionGCN, self).__init__()
        self.args = args
        self.is_gnn = True
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        gnn_layers_list = []
        num_incept = class_kwargs.get('num_incept', 5)

        for k, v in enumerate(range(len(layers_) - 2)):
            in_channels = layers_[k]
            out_channels = layers_[k+1]
            layer_list = []
            for i in range(num_incept):
                if k == 0:
                    layer_list.append(ChebConv(in_channels, out_channels, K=i + 1))
                else:
                    layer_list.append(ChebConv(in_channels*num_incept, out_channels, K=i + 1))
            gnn_layers_list.append(torch.nn.ModuleList(layer_list))
        self.gnn_layers = torch.nn.ModuleList(gnn_layers_list)
        self.out = nn.Linear(out_channels*num_incept, layers_[-1])

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        output_list = []

        for model in self.gnn_layers:
            layer_output = []
            for layer_ in model:
                cur_out = layer_(x, edge_idx, edge_weight=edge_wts)
                layer_output.append(cur_out)
            x = torch.cat(layer_output, -1)
        x = self.out(x)
        x = F.softmax(x, dim=-1)
        return x


class GAT(torch.nn.Module, ConfigMixin):
    """ N-layer Graph Attention Network. """

    def __init__(self, args):
        super(GAT, self).__init__()
        self.args = args
        self.is_gnn = True
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        heads_ = list(class_kwargs.get('heads'))
        gat_layers = []
        for k, v in enumerate(layers_[:-2]):
            if k == 0:
                cur_layer = GATConv(layers_[k], layers_[k+1],
                                    heads=heads_[k], concat=True)
            else:
                cur_layer = GATConv(layers_[k]*heads_[k], layers_[k+1],
                                    heads=heads_[k], concat=True)

            gat_layers.append(cur_layer)
        self.gat_layers = torch.nn.ModuleList(gat_layers)
        self.out = GATConv(layers_[-2]*heads_[k], layers_[-1], heads=heads_[k+1], concat=False)

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        for k, net in enumerate(self.gat_layers):
            x = net(x, edge_idx)
            x = F.relu(x)
        x = self.out(x, edge_idx)
        x = F.softmax(x, dim=-1)
        return x


class ChebNet(torch.nn.Module, ConfigMixin):
    """ N-layer ChebConv Network. """

    def __init__(self, args):
        super(ChebNet, self).__init__()
        self.args = args
        self.is_gnn = True
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        cheb_order = list(class_kwargs.get('K'))
        gnn_layers = []
        for k, v in enumerate(layers_[:-2]):
            cur_layer = ChebConv(layers_[k], layers_[k+1], K=cheb_order[k])
            gnn_layers.append(cur_layer)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.out = ChebConv(layers_[-2], layers_[-1], K=cheb_order[k+1])

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        for k, net in enumerate(self.gnn_layers):
            x = net(x, edge_idx)
            x = F.relu(x)
        x = self.out(x, edge_idx)
        x = F.softmax(x, dim=-1)
        return x


class ChebNetLinear(torch.nn.Module, ConfigMixin):
    """ N-layer ChebConv Network. """

    def __init__(self, args):
        super(ChebNetLinear, self).__init__()
        self.args = args
        self.is_gnn = True
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        cheb_order = list(class_kwargs.get('K'))
        gnn_layers = []
        for k, v in enumerate(layers_[:-2]):
            cur_layer = ChebConv(layers_[k], layers_[k+1], K=cheb_order[k])
            gnn_layers.append(cur_layer)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.out = torch.nn.Linear(layers_[-2], layers_[-1])

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        for k, net in enumerate(self.gnn_layers):
            x = net(x, edge_idx)
            x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x, dim=-1)
        return x


class GCNet(torch.nn.Module, ConfigMixin):
    """ N-layer Graph Convolutional Network. """

    def __init__(self, args):
        super(GCNet, self).__init__()
        self.args = args
        self.is_gnn = True
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        gnn_layers = []
        for k, v in enumerate(layers_[:-2]):
            cur_layer = tg.nn.GCNConv(layers_[k], layers_[k+1])
            gnn_layers.append(cur_layer)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.out = tg.nn.GCNConv(layers_[-2], layers_[-1])

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        for k, net in enumerate(self.gnn_layers):
            x = net(x, edge_idx)
            x = F.relu(x)
        x = self.out(x, edge_idx)
        x = F.softmax(x, dim=-1)
        return x


class GCNetLinear(torch.nn.Module, ConfigMixin, OutputLatentMixin):
    """ N-layer Graph Convolutional Network. """

    def __init__(self, args):
        super(GCNetLinear, self).__init__()
        self.args = args
        self.is_gnn = True
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        gnn_layers = []
        for k, v in enumerate(layers_[:-2]):
            cur_layer = tg.nn.GCNConv(layers_[k], layers_[k+1])
            gnn_layers.append(cur_layer)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.out = torch.nn.Linear(layers_[-2], layers_[-1])

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        for k, net in enumerate(self.gnn_layers):
            x = net(x, edge_idx)
            x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x, dim=-1)
        return x


class MaskedGCNet(torch.nn.Module, ConfigMixin):
    """ N-layer Graph Convolutional Network where mask matrix of observed entries is used as part of the feature
    matrix. """

    def __init__(self, args):
        super(MaskedGCNet, self).__init__()
        self.args = args
        self.is_gnn = True
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        gnn_layers = []
        for k, v in enumerate(layers_[:-2]):
            cur_layer = tg.nn.GCNConv(layers_[k], layers_[k+1])
            gnn_layers.append(cur_layer)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.out = tg.nn.GCNConv(layers_[-2], layers_[-1])

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = torch.cat([feat_matrix, mask_matrix.float()], 1)
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        for k, net in enumerate(self.gnn_layers):
            x = net(x, edge_idx)
            x = F.relu(x)
        x = self.out(x, edge_idx)
        x = F.softmax(x, dim=-1)
        return x


class MaskedInceptionGCN(torch.nn.Module, ConfigMixin):
    def __init__(self, args):
        super(MaskedInceptionGCN, self).__init__()
        self.args = args
        self.is_gnn = True
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        gnn_layers_list = []
        num_incept = class_kwargs.get('num_incept', 5)

        for k, v in enumerate(range(len(layers_) - 2)):
            in_channels = layers_[k]
            out_channels = layers_[k+1]
            layer_list = []
            for i in range(num_incept):
                if k == 0:
                    layer_list.append(ChebConv(in_channels, out_channels, K=i + 1))
                else:
                    layer_list.append(ChebConv(in_channels*num_incept, out_channels, K=i + 1))
            gnn_layers_list.append(torch.nn.ModuleList(layer_list))
        self.gnn_layers = torch.nn.ModuleList(gnn_layers_list)
        self.out = nn.Linear(out_channels*num_incept, layers_[-1])

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = torch.cat([feat_matrix, mask_matrix.float()], 1)
        adj_matrix = adj_matrix.float().sum(-1)
        edge_idx, edge_wts = tg.utils.dense_to_sparse(adj_matrix)
        output_list = []

        for model in self.gnn_layers:
            layer_output = []
            for layer_ in model:
                cur_out = layer_(x, edge_idx, edge_weight=edge_wts)
                layer_output.append(cur_out)
            x = torch.cat(layer_output, -1)
        x = self.out(x)
        x = F.softmax(x, dim=-1)
        return x


class CDGM(torch.nn.Module, ConfigMixin):
    """Continuous Differentiable Graph Module."""

    def __init__(self, args):
        super(CDGM, self).__init__()
        self.args = args
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        gnn_layers = []
        gl_layer = []
        for k, v in enumerate(layers_[:-1]):
            gl_ = nn.Linear(layers_[k], layers_[k + 1])
            gnn_ = nn.Linear(layers_[k], layers_[k + 1])
            gl_layer.append(gl_)
            gnn_layers.append(gnn_)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.gl_layers = torch.nn.ModuleList(gl_layer)
        self.temp = torch.nn.Parameter(
            torch.tensor(1e-1, requires_grad=True).cuda())
        self.theta = torch.nn.Parameter(
            torch.tensor(1e-1, requires_grad=True).cuda())

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        # adj_matrix = adj_matrix.sum(-1)
        # gnn_out = feat_matrix
        # gl_out =

        for k, (gl_model, gnn_model) in enumerate(zip(self.gl_layers, self.gnn_layers)):

            # For graph learning
            gl_out = F.relu(gl_model(x))
            diff = gl_out.unsqueeze(1) - gl_out.unsqueeze(0)
            diff = torch.pow(diff, 2).sum(2)
            mask_diff = diff != 0.0
            dist = -torch.sqrt(diff + torch.finfo(torch.float32).eps)
            dist = dist * mask_diff
            temp = 1.0 + self.temp
            theta = 5.0 + self.theta
            dist_ = temp * dist + theta
            adj = torch.sigmoid(dist_)
            # adj = adj * (adj >= 0.2)
            degree_mat = adj.sum(-1).diag()
            d_inv = degree_mat.inverse()
            # norm_adj = torch.eye(adj.shape[-1]).to(adj.device)
            # edge_idx, edge_wts = tg.utils.dense_to_sparse(adj)

            # Representation learning D^-1 A H_l W
            x = torch.mm(adj, gnn_model(x))
            x = torch.mm(d_inv, x)
            if k != (len(self.gnn_layers) - 1):
                x = F.relu(x)
        x = F.softmax(x, -1)
        return x


class CDGMLinear(torch.nn.Module, ConfigMixin):
    """Continuous Differentiable Graph Module with linear model at the last layer."""

    def __init__(self, args):
        super(CDGMLinear, self).__init__()
        self.args = args
        class_kwargs = self.load_config()
        layers_ = list(class_kwargs.get('layers'))
        latent_layers_ = list(class_kwargs.get('latent_layers'))
        gnn_layers = []
        gl_layer = []
        for k, v in enumerate(layers_[:-2]):
            # gl_ = nn.Linear(layers_[k], layers_[k + 1])
            gl_ = nn.Linear(layers_[k], latent_layers_[k])
            gnn_ = nn.Linear(layers_[k], layers_[k + 1])
            # gnn_ = tg.nn.GCNConv(layers_[k], layers_[k + 1])
            gl_layer.append(gl_)
            gnn_layers.append(gnn_)

        self.gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.gl_layers = torch.nn.ModuleList(gl_layer)
        self.out = torch.nn.Linear(layers_[-2], layers_[-1])

        self.temp = torch.nn.Parameter(
            torch.tensor(1e-1, requires_grad=True).cuda())
        self.theta = torch.nn.Parameter(
            torch.tensor(1e-1, requires_grad=True).cuda())

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        # adj_matrix = adj_matrix.sum(-1)
        # gnn_out = feat_matrix
        # gl_out =

        for k, (gl_model, gnn_model) in enumerate(zip(self.gl_layers, self.gnn_layers)):

            # For graph learning
            gl_out = F.relu(gl_model(x))
            # diff = gl_out.unsqueeze(1) - gl_out.unsqueeze(0)
            # diff = torch.pow(diff, 2).sum(2)
            # mask_diff = diff != 0.0
            # dist = -torch.sqrt(diff + torch.finfo(torch.float32).eps)
            # dist = dist * mask_diff

            adj = -utils.l2_mat(gl_out, gl_out)
            temp = 1.0 + self.temp
            theta = 5.0 + self.theta
            adj = torch.sigmoid(temp * adj + theta)
            # adj = adj * (adj >= 0.2)
            # norm_adj = torch.eye(adj.shape[-1]).to(adj.device)
            # edge_idx, edge_wts = tg.utils.dense_to_sparse(torch.sigmoid(temp * adj + theta))

            # Representation learning D^-1 A H_l W
            x = torch.mm(adj.sum(-1).diag().inverse(), torch.mm(adj, gnn_model(x)))
            # x = gnn_model(x, edge_idx, edge_weight=edge_wts)
            if k != (len(self.gnn_layers) - 1):
                x = F.relu(x)
        x = self.out(x)
        x = F.softmax(x, -1)
        return x


class MaskedMLPClassifier(nn.Module, ConfigMixin):
    """Multilayer Perceptrons with non-linear activation functions that uses the observed mask matrix as input."""

    def __init__(self, args):
        super(MaskedMLPClassifier, self).__init__()
        self.args = args
        class_kwargs = self.load_config()
        layers_list = list(class_kwargs.get('layers'))
        dropout_list = list(class_kwargs.get('p'))
        mlp_layers = []
        for k, v in enumerate(layers_list[:-2]):
            cur_layers_ = [
                nn.Linear(layers_list[k], layers_list[k + 1]),
                nn.Dropout(dropout_list[k]),
                nn.ReLU(),
                nn.BatchNorm1d(layers_list[k + 1])
            ]
            mlp_layers.extend(cur_layers_)

        self.enc = nn.Sequential(*mlp_layers)
        self.out = torch.nn.Linear(layers_list[-2], layers_list[-1])

    def forward(self, feature_matrix, adjacency_matrix, index, mask_matrix):
        x = torch.cat([feature_matrix, mask_matrix.float()], 1)
        enc_x = self.enc(x)
        pred_x = self.out(enc_x)
        pred_x = F.softmax(pred_x, dim=-1)
        return pred_x


class MGCN(torch.nn.Module, ConfigMixin):
    """Multi-graph GCN (https://link.springer.com/chapter/10.1007/978-3-030-32251-9_14)"""
    def __init__(self, args):
        super(MGCN, self).__init__()
        self.args = args
        gnn_list = []
        class_kwargs = self.load_config()

        # Initialize GNN for every graph
        gnn_layers = list(class_kwargs.get('layers'))
        for block in range(class_kwargs.num_graphs):
            gnn_list_layer_list = []
            # Initialize n layers of GNN
            for k, v in enumerate(gnn_layers[:-2]):
                gnn_list_layer_list.append(tg.nn.GCNConv(gnn_layers[k], gnn_layers[k+1]))
            gnn_list.append(torch.nn.ModuleList(gnn_list_layer_list))

        self.gnn_layers = torch.nn.ModuleList(gnn_list)
        self.rnn = torch.nn.LSTM(gnn_layers[-2], gnn_layers[-2], batch_first=True, bidirectional=True)
        self.rnn_out = torch.nn.Linear(gnn_layers[-2]*2, 1)
        self.out = torch.nn.Linear(gnn_layers[-2], gnn_layers[-1])

    def forward(self, feat_matrix, adj_matrix, get_item_index, set_index, val_index, mask_matrix):
        x = feat_matrix
        gnn_output = []
        for k, block in enumerate(self.gnn_layers):
            cur_adj = adj_matrix[:, :, k]
            edge_idx, edge_wts = tg.utils.dense_to_sparse(cur_adj)
            cur_out = x
            for model in block:
                cur_out = model(cur_out, edge_idx)
            gnn_output.append(cur_out)
        gnn_out = torch.stack(gnn_output, 1)

        # Get personalized attention scores
        rnn_hid, _ = self.rnn(gnn_out)
        rnn_out = torch.softmax(self.rnn_out(rnn_hid), 1)

        # Take weighted sum of every graph output using scores
        out = gnn_out * rnn_out
        out = out.sum(1)
        out = torch.softmax(self.out(out), -1)
        return out
