import torch
from sklearn.metrics import confusion_matrix, accuracy_score
from skorch import NeuralNetClassifier
import torch_geometric as tg



class GNNClassifier(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        ce_loss = super().get_loss(y_pred, y_true, *args, **kwargs)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # loss = nn.NLLLoss()
        # ce_loss = loss(y_pred, y_true.to(device))
        return ce_loss


class MultiLossClassifier(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        y_pred_cls, y_pred_feat, feat_gt = y_pred[0], y_pred[1], y_pred[2]
        # loss_weighting = torch.tensor([0.1727, 0.2806, 0.5467]).to(
        #     y_pred_cls.device)
        # nll_loss = torch.nn.NLLLoss(weight=loss_weighting)
        nll_loss = torch.nn.NLLLoss()
        log_softmax = torch.nn.LogSoftmax(dim=1)
        y_true = y_true.long().squeeze()
        ce_loss = nll_loss(log_softmax(y_pred_cls), y_true.to(y_pred[0].device))
        # rmse_loss = torch.nn.MSELoss()
        # mse_loss = rmse_loss(y_pred_feat, feat_gt)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # loss = nn.NLLLoss()
        # ce_loss = loss(y_pred, y_true.to(device))
        return ce_loss


class TransductiveMGMCModelWrapper(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """MGMC loss function"""

        # kwargs also contains the input from the forward method
        # y_pred contains the return value from forward method
        # pred_matrix = y_pred['pred_matrix']
        # z_bar = y_pred['z_delta_bar']  # pred matrix related to every graph
        # adj_matrix = y_pred['adj']

        pred_matrix = y_pred[0]
        z_bar = y_pred[1]  # pred matrix related to every graph
        adj_matrix = y_pred[2]
        train_idx = kwargs['X'][3][0]

        num_class = y_true.shape[-1]
        pred_features = pred_matrix[:, :-num_class]
        pred_target = pred_matrix[:, -num_class:]

        loss_weighting = None
        ce_loss_cls = torch.nn.CrossEntropyLoss(reduction='none',
                                                weight=loss_weighting)
        ce_loss = ce_loss_cls(pred_target, y_true.argmax(1).to(
            pred_target.device))
        ce_loss = ce_loss[train_idx].mean()

        # Dirichlet Norm
        dirichlet_loss = 0
        device = pred_target.device
        # graph_weights = [.1, .5, .1, .3]
        for i in range(adj_matrix.shape[-1]):
            cur_adj = adj_matrix[:, :, i]
            # cur_pred_feat = kwargs['X'][0].to(z_bar.device) + z_bar[:, :, i]
            # cur_pred_feat = cur_pred_feat[:, :-4]
            cur_pred_feat = z_bar[:, :, i]
            D_ = torch.pow(cur_adj.sum(1).float(), -0.5).diag()
            I = torch.eye(cur_adj.shape[0]).to(device)
            normalized_laplacian = I - torch.mm(torch.mm(D_.float(), cur_adj.float()), D_.float())
            res_matrix = torch.matmul(torch.matmul(cur_pred_feat.t().contiguous(), normalized_laplacian), cur_pred_feat)
            dirichlet_loss += torch.trace(res_matrix)

        # Squared-frobenius Norm
        frobenius_loss = 0
        mask_matrix = kwargs['X'][5].to(pred_features.device).float()
        gt_features = kwargs['X'][0].to(pred_features.device).float()[:, :-num_class]
        squared_diff = ((gt_features - pred_features)**2) * mask_matrix
        frobenius_loss += squared_diff.sum()

        # ce_weight = 1.0
        # dirichlet_weight = .000001
        # frob_weight = .0001
        ce_weight = 1.0
        dirichlet_weight = .000001
        frob_weight = .0001
        total_loss = ce_weight*ce_loss + dirichlet_weight*dirichlet_loss + \
                     frob_weight*frobenius_loss
        # total_loss = ce_loss

        ### Confusion matrix
        print('Training Confusion Matrix')
        ce_loss_cls(pred_target, y_true.argmax(1).to(pred_target.device))
        train_pred = pred_target[train_idx].detach().cpu().numpy().argmax(-1)
        train_gt = y_true[train_idx].numpy().argmax(-1)
        print(confusion_matrix(train_gt, train_pred))

        print('Training Cross-entropy loss:', ce_loss)
        print('Training Dirichlet loss:', dirichlet_loss)
        print('Trianing Frobenius loss:', frobenius_loss)

        return total_loss


class TransductiveClassificationWrapper(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """Transductive classification loss for GNNs."""

        # kwargs also contains the input from the forward method
        # y_pred contains the return value from forward method
        # pred_matrix = y_pred['pred_matrix']
        # z_bar = y_pred['z_delta_bar']  # pred matrix related to every graph
        # adj_matrix = y_pred['adj']


        # Classification Loss
        train_idx = kwargs['X']['set_index'][0]
        loss_weighting = None
        ce_loss_cls = torch.nn.CrossEntropyLoss(reduction='none',
                                                weight=loss_weighting)
        ce_loss = ce_loss_cls(y_pred, y_true.argmax(1).to(y_pred.device))
        ce_loss = ce_loss[train_idx].mean()

        ### Confusion matrix
        print('Training Confusion Matrix')
        train_pred = y_pred[train_idx].detach().cpu().numpy().argmax(-1)
        train_gt = y_true[train_idx].numpy().argmax(-1)
        print(confusion_matrix(train_gt, train_pred))
        print('Training Cross-entropy loss:', ce_loss)
        return ce_loss


class TransductiveMGRGCNNModelWrapper(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """MG-RGCNN loss function"""

        # kwargs also contains the input from the forward method
        # y_pred contains the return value from forward method
        # pred_matrix = y_pred['pred_matrix']
        # z_bar = y_pred['z_delta_bar']  # pred matrix related to every graph
        # adj_matrix = y_pred['adj']

        pred_matrix = y_pred[0]
        adj_matrix = y_pred[1]
        train_idx = kwargs['X'][3][0]

        num_class = y_true.shape[-1]
        pred_features = pred_matrix[:, :-num_class]
        pred_target = pred_matrix[:, -num_class:]

        # Classification Loss
        loss_weighting = None
        ce_loss_cls = torch.nn.CrossEntropyLoss(reduction='none',
                                                weight=loss_weighting)
        ce_loss = ce_loss_cls(pred_target, y_true.argmax(1).to(pred_target.device))
        ce_loss = ce_loss[train_idx].mean()

        # Squared-frobenius Norm
        frobenius_loss = 0
        mask_matrix = kwargs['X'][5].to(pred_features.device).float()
        gt_features = kwargs['X'][0].to(pred_features.device).float()[:,:-num_class]
        squared_diff = ((gt_features - pred_features)**2) * mask_matrix
        frobenius_loss += squared_diff.sum()

        # Dirichlet Norm
        dirichlet_loss = 0
        device = pred_target.device
        for i in range(adj_matrix.shape[-1]):
            cur_adj = adj_matrix[:, :, i]
            # cur_pred_feat = kwargs['X'][0].to(z_bar.device) + z_bar[:, :, i]
            # cur_pred_feat = cur_pred_feat[:, :-4]
            cur_pred_feat = gt_features
            D_ = torch.pow(cur_adj.sum(1).float(), -0.5).diag()
            I = torch.eye(cur_adj.shape[0]).to(device)
            normalized_laplacian = I - torch.mm(torch.mm(D_.float(), cur_adj.float()), D_.float())
            res_matrix = torch.matmul(torch.matmul(cur_pred_feat.t().contiguous(), normalized_laplacian), cur_pred_feat)
            dirichlet_loss += torch.trace(res_matrix)

        ce_weight = self.module_.args.ce_wt
        dirichlet_weight = self.module_.args.dirch_wt
        frob_weight = self.module_.args.frob_wt

        # ce_weight = 1.0
        # dirichlet_weight = .000001
        # frob_weight = .0001
        total_loss = ce_weight*ce_loss + dirichlet_weight*dirichlet_loss + frob_weight*frobenius_loss

        ### Confusion matrix
        print('Training Confusion Matrix')
        ce_loss_cls(pred_target, y_true.argmax(1).to(pred_target.device))
        train_pred = pred_target[train_idx].detach().cpu().numpy().argmax(-1)
        train_gt = y_true[train_idx].numpy().argmax(-1)
        print(confusion_matrix(train_gt, train_pred))

        print('Training Cross-entropy loss:', ce_loss)
        print('Training Dirichlet loss:', dirichlet_loss)
        print('Trianing Frobenius loss:', frobenius_loss)

        return total_loss


class TransductiveMLPMCWrapper(NeuralNetClassifier):
    def get_loss(self, y_pred, y_true, *args, **kwargs):
        """MLP Matrix Completion loss function"""

        # kwargs also contains the input from the forward method
        # y_pred contains the return value from forward method
        # pred_matrix = y_pred['pred_matrix']
        # z_bar = y_pred['z_delta_bar']  # pred matrix related to every graph
        # adj_matrix = y_pred['adj']

        pred_matrix = y_pred[0]
        # adj_matrix = y_pred[1]
        train_idx = kwargs['X'][3][0]

        num_class = y_true.shape[-1]
        pred_features = pred_matrix[:, :-num_class]
        pred_target = pred_matrix[:, -num_class:]

        # Classification Loss
        loss_weighting = None
        ce_loss_cls = torch.nn.CrossEntropyLoss(reduction='none', weight=loss_weighting)
        ce_loss = ce_loss_cls(pred_target, y_true.argmax(1).to(pred_target.device))
        ce_loss = ce_loss[train_idx].mean()

        # Squared-frobenius Norm
        frobenius_loss = 0
        mask_matrix = kwargs['X'][5].to(pred_features.device).float()
        gt_features = kwargs['X'][0].to(pred_features.device).float()[:, :-num_class]
        squared_diff = ((gt_features - pred_features)**2) * mask_matrix
        frobenius_loss += squared_diff.sum()

        ce_weight = 100.0
        frob_weight = .0001
        total_loss = ce_weight*ce_loss + frob_weight*frobenius_loss
        # Confusion matrix
        print('Training Confusion Matrix')
        ce_loss_cls(pred_target, y_true.argmax(1).to(pred_target.device))
        train_pred = pred_target[train_idx].detach().cpu().numpy().argmax(-1)
        train_gt = y_true[train_idx].numpy().argmax(-1)
        print(confusion_matrix(train_gt, train_pred))

        print('Training Cross-entropy loss:', ce_loss)
        print('Trianing Frobenius loss:', frobenius_loss)

        return total_loss
