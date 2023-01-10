import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch import nn


class SAGE(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_channels=178,
                 num_layers=3,
                 dropout=0.0,
                 batchnorm=False):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_feats, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fc = SAGEConv(hidden_channels, out_feats)

        self.dropout = dropout

    def forward(self, x, edge_index, *args, **kwargs):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x, edge_index)


def sage(in_feats, out_feats, **kwargs):
    return SAGE(in_feats, out_feats, **kwargs)


def sage_bn(in_feats, out_feats, **kwargs):
    return SAGE(in_feats, out_feats, batchnorm=True, **kwargs)


def sage_bn_drop05(in_feats, out_feats, **kwargs):
    return SAGE(in_feats, out_feats, batchnorm=True, dropout=0.05, **kwargs)


def sage_bn_drop10(in_feats, out_feats, **kwargs):
    return SAGE(in_feats, out_feats, batchnorm=True, dropout=0.1, **kwargs)

def sage_bn_drop30(in_feats, out_feats, **kwargs):
    return SAGE(in_feats, out_feats, batchnorm=True, dropout=0.3, **kwargs)

def sage_bn_drop50(in_feats, out_feats, **kwargs):
    return SAGE(in_feats, out_feats, batchnorm=True, dropout=0.5, **kwargs)