import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_channels=128,
                 dropout=0.0,
                 layer_heads=[2, 2, 1],
                 batchnorm=False):
        super(GAT, self).__init__()
        num_layers = len(layer_heads)
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATConv(in_feats,
                    hidden_channels,
                    heads=layer_heads[0],
                    concat=True))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(
                torch.nn.BatchNorm1d(hidden_channels * layer_heads[0]))
        for i in range(1, num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels * layer_heads[i - 1],
                        hidden_channels,
                        heads=layer_heads[i],
                        concat=True))
            if self.batchnorm:
                self.bns.append(
                    torch.nn.BatchNorm1d(hidden_channels * layer_heads[i - 1]))
        self.fc = GATConv(hidden_channels * layer_heads[num_layers - 2],
                          out_feats,
                          heads=layer_heads[num_layers - 1],
                          concat=False)

        self.dropout = dropout

    def forward(self, x, edge_index, *args, **kwargs):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc(x, edge_index)


def gat(in_feats, out_feats, **kwargs):
    return GAT(in_feats, out_feats, **kwargs)


def gat_bn(in_feats, out_feats, **kwargs):
    return GAT(in_feats, out_feats, batchnorm=True, **kwargs)


def gat_bn_drop05(in_feats, out_feats, **kwargs):
    return GAT(in_feats, out_feats, batchnorm=True, dropout=0.05, **kwargs)


def gat_bn_drop10(in_feats, out_feats, **kwargs):
    return GAT(in_feats, out_feats, batchnorm=True, dropout=0.1, **kwargs)

def gat_bn_drop30(in_feats, out_feats, **kwargs):
    return GAT(in_feats, out_feats, batchnorm=True, dropout=0.3, **kwargs)
