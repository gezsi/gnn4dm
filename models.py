import torch

import torch_geometric

import torch.nn.functional as F
from torch.nn import Parameter
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv


import math

from utils import cosine_self_similarity, inv_softplus

EPS = 1e-15

def reset(value):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)

class PositiveLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, ids: list = None, 
                 device = None, dtype = None):
        super(PositiveLinear, self).__init__(in_features, out_features, bias, device, dtype)

        assert out_features == len(ids), "Number of ids doesn't match with the number of out_features."
        
        # populate mappings of identifiers to indices of the weight parameter matrix
        self.id_list = ids
        self.id_to_position = {value: index for index, value in enumerate(ids)} if ids is not None else None

        if ids != None:
            assert len(ids) == out_features, "The number of ids doesn't match with the number of output features."

    def forward(self, x):
        self.weight.data.clamp_(min=0.0)  # Apply the constraint on weights
        return super(PositiveLinear, self).forward(x)

    def l1_l2_losses(self):
        positive_weights = torch.where(self.weight > 0, self.weight, torch.zeros_like(self.weight))

        l1 = torch.norm(positive_weights, p=1)
        l2 = torch.norm(positive_weights, p=2)

        return l1, l2

    
class GNN(torch.nn.Module):
    def __init__(self, 
                 in_channels, hidden_channels_before_module_representation, module_representation_channels, out_models : dict, 
                 dropout = 0.0, batchnorm = False, transform_probability_method : str = "tanh", threshold = 1.0,
                 type : str = "GCN" ):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.batchnorms_convs = torch.nn.ModuleList()
        self.output_models = torch.nn.ModuleDict({ key: out_model for key, out_model in out_models.items() })

        self.type = type

        # clean zeros from hidden channel counts
        if 0 in hidden_channels_before_module_representation: hidden_channels_before_module_representation.remove(0)
        
        # 1. GNN layers from input to module_representation
        channels = in_channels
        for hidden_channel in hidden_channels_before_module_representation:
            if type == "GCN":
                self.convs.append( torch_geometric.nn.GCNConv(channels, hidden_channel, improved=True, cached=True) )
            elif type == "MLP":
                self.convs.append( torch.nn.Linear(channels, hidden_channel) )
            elif type == "GAT":
                self.convs.append( torch_geometric.nn.GATv2Conv(channels, hidden_channel) )
            elif type == "SAGE":
                self.convs.append( torch_geometric.nn.SAGEConv(channels, hidden_channel) )
            if batchnorm:
                self.batchnorms_convs.append( torch.nn.BatchNorm1d(hidden_channel) )
            channels = hidden_channel

        if type == "GCN":
            self.conv_last = torch_geometric.nn.GCNConv(channels, module_representation_channels, improved=True, cached=True)
        elif type == "MLP":
            self.conv_last = torch.nn.Linear(channels, module_representation_channels)
        elif type == "GAT":
            self.conv_last = torch_geometric.nn.GATv2Conv(channels, module_representation_channels)
        elif type == "SAGE":
            self.conv_last = torch_geometric.nn.SAGEConv(channels, module_representation_channels)

        channels = module_representation_channels
        
        self.dropout = dropout
        
        self.transform_probability_method = transform_probability_method
        if threshold == 'auto':
            if self.transform_probability_method == 'tanh':
                self.threshold = torch.nn.Parameter(torch.tensor(inv_softplus(1.0)), requires_grad = True)
            else:
                self.threshold = torch.nn.Parameter(torch.tensor(inv_softplus(0.5)), requires_grad = True)
        else:
            self.threshold = torch.nn.Parameter(torch.tensor(inv_softplus(threshold)), requires_grad = False)

        self._short_name = f"{self.type}_{in_channels}-[{'-'.join([str(h) for h in hidden_channels_before_module_representation])}]--Modules:{module_representation_channels}--LIN[{'|'.join([k for k in out_models])}"

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
            if self.type == "GCN": 
                torch.nn.init.orthogonal_( conv.lin.weight )        # orthogonal initialization
        self.conv_last.reset_parameters()
        if self.type == "GCN":
            torch.nn.init.orthogonal_( self.conv_last.lin.weight )  # orthogonal initialization
        for ll in self.output_models.values():
            ll.reset_parameters()
        for bn in self.batchnorms_convs:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            if self.dropout != 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.type == "MLP":
                x = self.convs[i](x) 
            else:
                x = self.convs[i](x, edge_index) 
            x = F.relu(x)
            if i < len(self.batchnorms_convs):
                x = self.batchnorms_convs[i](x)
            
        if self.dropout != 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.type == "MLP":
            x = self.conv_last(x)
        else:
            x = self.conv_last(x, edge_index)

        # computing the module representations by applying a softplus activation (to enforce non-negative values)
        x = F.softplus(x)
        # this leads to the module representations which will be returned
        module_representation = x

        # transforming the module representations into probabilities 
        # x = tanh( k * x )
        # where k is a learnable scaling parameter
        # tanh(k*x) > 0.5 <=> x > 0.549306/k
        if self.transform_probability_method == 'tanh':
            x = torch.tanh( F.softplus(self.threshold) * x )
        else:
            x = x / (x + F.softplus(self.threshold))

        # applying the final "interpretable" modules
        y_preds = dict()
        for key in self.output_models:
            y_preds[key] = self.output_models[key](x)

        return module_representation, y_preds
    
    def output_models_l1_l2_losses(self):
        l1_positives_loss = 0.0
        l2_positives_loss = 0.0

        for model in self.output_models.values():
            l1, l2 = model.l1_l2_losses()          
            l1_positives_loss += l1 
            l2_positives_loss += l2 

        return l1_positives_loss, l2_positives_loss
    
    @property
    def name(self):
        return self._short_name

    def __str__(self):
        arch = ""
        for i in range(len(self.convs)):
            if self.dropout != 0.0:
                arch += f"(dropout:{self.dropout}); "
            arch += self.convs[i].__repr__() + "; "
            arch += "ReLU; "
            if i < len(self.batchnorms_convs):
                arch += self.batchnorms_convs[i].__repr__() + "; "
        if self.dropout != 0.0:
            arch += f"(dropout:{self.dropout}); "
        arch += self.conv_last.__repr__() + "; "
        arch += "SoftPlus; "
        arch += "{" + '|'.join( [self.output_models[key].__repr__() for key in self.output_models] ) + "}"
        return arch


class InnerProductDecoder(torch.nn.Module):
    r"""Inner product decoder.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def __init__(self):
        super().__init__()
        self._short_name = "IP"

    def forward(self, z: torch.Tensor, source: torch.Tensor, destination: torch.Tensor, sigmoid: bool = False) -> torch.Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[source] * z[destination]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_full(self, z: torch.Tensor, sigmoid: bool = False) -> torch.Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`False`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
    
    @property
    def name(self):
        return self._short_name

    def __str__(self):
        return f"InnerProductDecoder"

class GAEL(torch.nn.Module):
    r"""Graph Autoencoder for Link Community Prediction.

    Args:
        encoder (Module): The encoder module.
        decoder (Module): The decoder module. 
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bceloss = torch.nn.BCEWithLogitsLoss()
        GAEL.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)
    
    def getThresholdOfCommunities(self):
        r"""Returns the threshold above which the probability of being in a module is greater than 0.5."""
        if self.encoder.transform_probability_method == 'tanh':
            return 0.549306 / F.softplus(self.encoder.threshold)
        else:
            return F.softplus(self.encoder.threshold)

    def edgeTriplet_loss(self, L_pos, pos_index, neg_index, margin=0.5):
        r"""Computes triplet loss for anchor, positive and negative (but existing) edges."""
        
        assert L_pos.shape[0] == len(pos_index), "The length of positive edge indices does not match the number of anchor edges."
        assert L_pos.shape[0] == len(neg_index), "The length of negative edge indices does not match the number of anchor edges."
        
        E_anchor = torch.exp( -L_pos )
        E_positive = E_anchor[pos_index,:]
        E_negative = E_anchor[neg_index,:]

        probability_pos_edges = 1-torch.prod(E_anchor + E_positive - E_anchor*E_positive, dim=1)
        probability_neg_edges = 1-torch.prod(E_anchor + E_negative - E_anchor*E_negative, dim=1)

        prob_diff = probability_neg_edges - probability_pos_edges + margin

        losses = torch.max( prob_diff, torch.zeros_like(prob_diff) )
        return torch.mean(losses)
    
    def bce_loss(self, y_pred, target):
        return self.bceloss( y_pred, target )
    
    def cosine_similarity_loss(self, F, type : str, min_threshold : float = 0.0):
        valid_type_values = ['max','l1','l2']

        if type not in valid_type_values:
            raise ValueError(f"Invalid 'type' argument. Must be one of: {', '.join(valid_type_values)}")
    
        # Calculate cosine similarities for each column of the matrix
        cosine_similarities = cosine_self_similarity(F)

        eye_matrix = torch.eye(cosine_similarities.size(0), device=cosine_similarities.device)
        off_diagonal_clamped_cosine_similarities = torch.clamp( ( cosine_similarities - eye_matrix) - min_threshold, min = 0.0 )

        if type == 'max':
            loss = off_diagonal_clamped_cosine_similarities.max()
        elif type == 'l1':
            loss = off_diagonal_clamped_cosine_similarities.sum()
        elif type == 'l2':
            eye_matrix = torch.eye(cosine_similarities.size(0), device=cosine_similarities.device)
            loss = torch.norm(off_diagonal_clamped_cosine_similarities, p=2)

        return loss

    def rmse_of_module_size_loss(self, F, threshold, expected_mean):
        """
        Calculates the root mean squared error of the observed module sizes and the expected mean module size.
        """
        # Calculate the column-wise count of elements greater than the threshold
        observed_values = torch.sum(F >= threshold, dim=0, dtype=torch.float)
        observed_values = observed_values[ observed_values > 0.0 ]
        # Calculate root mean squared error of the observed module sizes and the expected mean module size
        rmse = torch.sqrt(torch.mean((observed_values - expected_mean)**2))
        return rmse

    def gcn_l1_l2_losses(self):
        l1_loss = 0.0
        l2_loss = 0.0

        for layer in self.encoder.modules():
            if isinstance(layer, torch_geometric.nn.GCNConv):
                l1_loss += torch.norm(layer.lin.weight, p=1)
                l2_loss += torch.norm(layer.lin.weight, p=2)

        return l1_loss, l2_loss
    
    def output_models_l1_l2_losses(self):
        return self.encoder.output_models_l1_l2_losses()
    
    def nll_BernoulliPoisson_loss(self, strength_pos_edges, strength_neg_edges, epsilon = 1e-8):
        r"""Given latent variables :obj:`H`, computes the Bernoulli-Poisson 
        loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges :obj:`neg_edge_index`.

        Args:
            H (Tensor): The latent space :math:`\mathbf{H}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor): The negative edges to train against.
        """
        
        ll_pos_edges = -torch.mean( torch.log( -torch.expm1( -strength_pos_edges - epsilon ) ) )
        ll_neg_edges = torch.mean( strength_neg_edges )

        ll = (ll_pos_edges + ll_neg_edges) / 2.0
        return ll
    
    def nll_BernoulliPoisson_loss_full(self, H, L_pos, num_edges, num_nonedges, epsilon = 1e-8):
        """Compute full loss."""
        strength_pos_edges = torch.sum( L_pos, dim=1 )
        strength_all_possible_edges = torch.sum( self.decoder.compute_all_possible_edge_strengths( H ) )
        loss_nonedges = strength_all_possible_edges - torch.sum( strength_pos_edges )

        ll_pos_edges = -torch.sum( torch.log( -torch.expm1( -strength_pos_edges - epsilon ) ) )

        # if self.balance_loss:
        #     neg_scale = 1.0
        # else:
        #     neg_scale = num_nonedges / num_edges
        # ll = (ll_pos_edges / num_edges + neg_scale * loss_nonedges / num_nonedges) / (1 + neg_scale)
        ll = (ll_pos_edges / num_edges + loss_nonedges / num_nonedges)
        return ll

    def configure_optimizers(self, params):
        """
        This implementation is based on https://github.com/karpathy/minGPT
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, PositiveLinear, torch_geometric.nn.GCNConv, torch_geometric.nn.GATv2Conv, torch_geometric.nn.SAGEConv)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('att') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif fpn == 'encoder.threshold':
                    # special cases as not decayed
                    no_decay.add(fpn)
                elif 'adjacency_tensor_weights' in pn:
                    # special cases as not decayed
                    no_decay.add(fpn)

        #print( f"decay: {str(decay)}")
        #print( f"no_decay: {str(no_decay)}")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": params.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=params.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.learning_rate_decay_step_size, gamma=0.85)

        return optimizer, scheduler

    @property
    def name(self):
        return f"{self.encoder.name}--{self.decoder.name}"

    def __str__(self):
        return f"GAEL( Encoder: {str(self.encoder)}, Decoder: {str(self.decoder)} )"

