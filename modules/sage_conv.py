"""Torch Module for GraphSAGE layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import functional as F
from dgl.nn.pytorch.softmax import edge_softmax

import dgl.function as fn
from dgl.utils import expand_as_pair, check_eq_shape


class SAGEConv(nn.Module):
    r"""GraphSAGE layer from paper `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__.

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} & = \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} & = \sigma \left(W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{l+1} + b) \right)

        h_{i}^{(l+1)} & = \mathrm{norm}(h_{i}^{l})

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.

        If aggregator type is ``gcn``, the feature size of source and destination nodes
        are required to be the same.
    out_feats : int
        Output feature size.
    feat_drop : float
        Dropout rate on features, default: ``0``.
    aggregator_type : str
        Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._edge_feats = edge_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        # self.attn_drop = nn.Dropout(0.2)
        # self.leaky_relu = nn.LeakyReLU(0.2)
        # weight = True
        # if weight:
        #     self.weight = nn.Parameter(th.Tensor(edge_feats, 1))

        # aggregator type: mean/pool/lstm/gcn
        if 'pool' in aggregator_type:
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        # if aggregator_type == 'poolmean':
        #     self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats * edge_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats * edge_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        # if self.weight is not None:
        #     nn.init.xavier_uniform_(self.weight)
        if 'pool' in self._aggre_type:
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m']  # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, graph, feat, weight=None):
        r"""Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()

        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)

        h_self = feat_dst

        # if weight is None:
        #     weight = self.weight
        #     if 'w' in graph.edata:
        #         graph.edata['w'] = th.matmul(graph.edata['w'], weight)
        #         e = self.leaky_relu(graph.edata.pop('w'))
        #         # compute softmax
        #         graph.edata['a'] = self.attn_drop(F.softmax(e))

        if self._aggre_type == 'mean':
            graph.srcdata['h'] = feat_src
            if 'a' in graph.edata:
                graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.mean('m', 'neigh'))
            else:
                graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'gcn':
            check_eq_shape(feat)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst  # same as above if homogeneous
            if 'a' in graph.edata:
                graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.sum('m', 'neigh'))
                h_neigh = graph.dstdata['neigh'] + graph.dstdata['h']
            else:
                graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
                degs = graph.in_degrees().to(feat_dst)  # divide in_degrees
                h_neigh = (graph.dstdata['neigh'] + (graph.dstdata['h'])) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            if 'a' in graph.edata:
                graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.max('m', 'neigh'))
            else:
                graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'poolmean':
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            if 'a' in graph.edata:
                graph.update_all(fn.u_mul_e('h', 'a', 'm'), fn.mean('m', 'neigh'))
            else:
                graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'lstm':
            graph.srcdata['h'] = feat_src
            graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            h_neigh = graph.dstdata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            h_neigh = h_neigh.flatten(1)
            rst = self.fc_neigh(h_neigh)
        else:
            h_self = h_self.flatten(1)
            h_neigh = h_neigh.flatten(1)
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst
